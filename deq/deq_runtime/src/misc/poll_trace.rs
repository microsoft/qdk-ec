//! Sampled wake-to-poll diagnostics. Pending input time is not runnable-queue time.

use crate::decoder::worker_trace::{WorkerTrace, sampled_pipeline_gid};
use futures_util::task::AtomicWaker;
use std::collections::BTreeSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

struct WakeStamp {
    time: Instant,
    cpu: i32,
    self_wake: bool,
}

#[derive(Default)]
struct WakeState {
    first: Option<WakeStamp>,
    polling: Option<ThreadId>,
    calls: u64,
    self_calls: u64,
}

#[derive(Default)]
struct WakeProbe {
    parent: AtomicWaker,
    state: Mutex<WakeState>,
}

impl Wake for WakeProbe {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        {
            let mut state = self.state.lock().unwrap();
            let self_wake = state.polling == Some(std::thread::current().id());
            state.calls += 1;
            state.self_calls += u64::from(self_wake);
            state.first.get_or_insert_with(|| WakeStamp {
                time: Instant::now(),
                cpu: current_cpu(),
                self_wake,
            });
        }
        self.parent.wake();
    }
}

fn current_cpu() -> i32 {
    #[cfg(target_os = "linux")]
    {
        // sched_getcpu is thread-local observation; -1 records an unavailable observation.
        unsafe { libc::sched_getcpu() }
    }
    #[cfg(not(target_os = "linux"))]
    {
        -1
    }
}

struct PollStatistics {
    trace: WorkerTrace,
    probe: Arc<WakeProbe>,
    polls: u64,
    wake_polls: u64,
    poll_wall: Duration,
    wake_delay: Duration,
    max_wake_delay: Duration,
    self_delay: Duration,
    poll_cpus: BTreeSet<i32>,
    wake_cpus: BTreeSet<i32>,
    completed: bool,
}

impl PollStatistics {
    fn new(trace: WorkerTrace) -> Self {
        Self {
            trace,
            probe: Arc::default(),
            polls: 0,
            wake_polls: 0,
            poll_wall: Duration::ZERO,
            wake_delay: Duration::ZERO,
            max_wake_delay: Duration::ZERO,
            self_delay: Duration::ZERO,
            poll_cpus: BTreeSet::new(),
            wake_cpus: BTreeSet::new(),
            completed: false,
        }
    }
}

impl Drop for PollStatistics {
    fn drop(&mut self) {
        // A retained child waker must not keep its parent task alive after cancellation.
        drop(self.probe.parent.take());
        let state = self.probe.state.lock().unwrap();
        self.trace.property("polls", || self.polls.to_string());
        self.trace.property("wake_calls", || state.calls.to_string());
        self.trace.property("self_wake_calls", || state.self_calls.to_string());
        self.trace.property("wake_polls", || self.wake_polls.to_string());
        self.trace.property("poll_wall_ns", || self.poll_wall.as_nanos().to_string());
        self.trace
            .property("wake_delay_ns", || self.wake_delay.as_nanos().to_string());
        self.trace
            .property("max_wake_delay_ns", || self.max_wake_delay.as_nanos().to_string());
        self.trace
            .property("self_wake_delay_ns", || self.self_delay.as_nanos().to_string());
        self.trace.property("poll_cpus", || format!("{:?}", self.poll_cpus));
        self.trace.property("wake_cpus", || format!("{:?}", self.wake_cpus));
        self.trace.property("completed", || self.completed.to_string());
    }
}

pub(crate) struct TracedPolls<F> {
    future: Pin<Box<F>>,
    statistics: Option<Box<PollStatistics>>,
}

impl<F> TracedPolls<F> {
    pub(crate) fn new(future: Pin<Box<F>>, gid: u64) -> Self {
        let statistics = if crate::misc::fastrace::system_diagnostics_enabled() && sampled_pipeline_gid(gid) {
            let mut trace = WorkerTrace::root("gadget_polls", true);
            trace.property("gid", || gid.to_string());
            trace.property("sample_rate", || "64".into());
            trace.property("cpu_observation", || {
                if cfg!(target_os = "linux") {
                    "sched_getcpu"
                } else {
                    "unavailable"
                }
                .into()
            });
            Some(Box::new(PollStatistics::new(trace)))
        } else {
            None
        };
        Self { future, statistics }
    }
}

impl<F: Future> Future for TracedPolls<F> {
    type Output = F::Output;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let Some(statistics) = &mut this.statistics else {
            return this.future.as_mut().poll(context);
        };
        let entered = Instant::now();
        statistics.probe.parent.register(context.waker());
        let wake = {
            let mut state = statistics.probe.state.lock().unwrap();
            state.polling = Some(std::thread::current().id());
            state.first.take()
        };
        if let Some(wake) = wake {
            let delay = entered.saturating_duration_since(wake.time);
            statistics.wake_polls += 1;
            statistics.wake_delay += delay;
            statistics.max_wake_delay = statistics.max_wake_delay.max(delay);
            if wake.self_wake {
                statistics.self_delay += delay;
            }
            statistics.wake_cpus.insert(wake.cpu);
        }
        statistics.poll_cpus.insert(current_cpu());
        if statistics.polls == 0 {
            statistics.trace.event_at("first_poll", entered);
        }
        statistics.polls += 1;
        let waker = Waker::from(Arc::clone(&statistics.probe));
        let started = Instant::now();
        let result = this.future.as_mut().poll(&mut Context::from_waker(&waker));
        statistics.poll_wall += started.elapsed();
        statistics.probe.state.lock().unwrap().polling = None;
        if result.is_ready() {
            statistics.completed = true;
            drop(statistics.probe.parent.take());
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Default)]
    struct WakeCounter(AtomicUsize);

    impl Wake for WakeCounter {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[test]
    fn forwarded_wakes_preserve_results_and_release_parent_references() {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let mut future = TracedPolls {
            future: Box::pin(receiver),
            statistics: Some(Box::new(PollStatistics::new(WorkerTrace::root("test", false)))),
        };
        let probe = Arc::downgrade(&future.statistics.as_ref().unwrap().probe);
        let counter = Arc::new(WakeCounter::default());
        let waker = Waker::from(Arc::clone(&counter));
        let mut context = Context::from_waker(&waker);
        assert!(Pin::new(&mut future).poll(&mut context).is_pending());
        sender.send(7).unwrap();
        assert_eq!(counter.0.load(Ordering::Relaxed), 1);
        let Poll::Ready(result) = Pin::new(&mut future).poll(&mut context) else {
            panic!("forwarded wake must make the result available");
        };
        assert_eq!(result.unwrap(), 7);
        let statistics = future.statistics.as_ref().unwrap();
        assert_eq!(statistics.polls, 2);
        assert_eq!(statistics.wake_polls, 1);
        drop(future);
        assert!(probe.upgrade().is_none());
        assert_eq!(Arc::strong_count(&counter), 2);
    }

    #[test]
    fn retained_child_wakers_cannot_retain_or_wake_a_cancelled_parent() {
        let stored = Arc::new(Mutex::new(None));
        let capture = Arc::clone(&stored);
        let mut future = TracedPolls {
            future: Box::pin(std::future::poll_fn(move |context| {
                *capture.lock().unwrap() = Some(context.waker().clone());
                Poll::<()>::Pending
            })),
            statistics: Some(Box::new(PollStatistics::new(WorkerTrace::root("test", false)))),
        };
        let counter = Arc::new(WakeCounter::default());
        let waker = Waker::from(Arc::clone(&counter));
        assert!(Pin::new(&mut future).poll(&mut Context::from_waker(&waker)).is_pending());
        assert_eq!(Arc::strong_count(&counter), 3);
        drop(future);
        assert_eq!(Arc::strong_count(&counter), 2);
        let calls = counter.0.load(Ordering::Relaxed);
        stored.lock().unwrap().take().unwrap().wake();
        assert_eq!(counter.0.load(Ordering::Relaxed), calls);
    }
}
