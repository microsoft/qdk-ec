//! Monotonic input deadlines without async timer-wheel rounding.

use std::collections::BTreeMap;
use std::io;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;

#[derive(Default)]
struct State {
    pending: BTreeMap<(Instant, u64), oneshot::Sender<()>>,
    sequence: u64,
    stopping: bool,
}

#[derive(Default)]
struct Shared {
    state: Mutex<State>,
    changed: Condvar,
}

/// A sleeping thread serving cancellable, best-effort precise deadlines.
///
/// Async callers wake before their deadline and yield through a bounded precision window.
/// OS scheduling can still deliver late; a deadline is never delivered early.
pub struct DeadlineTimer {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

impl DeadlineTimer {
    /// Maximum precision window per deadline; async emitters yield instead of blocking.
    pub const BLOCKING_PRECISION_WINDOW: Duration = Duration::from_micros(500);

    /// Wait on a dedicated caller thread without an additional async handoff.
    ///
    /// Do not call this from an async runtime worker; use [`Self::wait_until`] there.
    /// The final bounded precision window avoids OS sleep coalescing. This costs
    /// at most 0.5 ms of CPU time per deadline, not a continuously spinning worker.
    pub fn wait_until_blocking(deadline: Instant) {
        Self::wait_until_with_work(deadline, || false);
    }

    /// Serve available work while waiting on a caller-owned thread.
    ///
    /// `work` returns whether it performed work. Notifications can interrupt the
    /// parked wait, allowing a single caller worker to keep dependencies moving.
    pub fn wait_until_with_work(deadline: Instant, mut work: impl FnMut() -> bool) {
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return;
            }
            if remaining > Self::BLOCKING_PRECISION_WINDOW {
                if work() {
                    continue;
                }
                std::thread::park_timeout(remaining.saturating_sub(Self::BLOCKING_PRECISION_WINDOW));
            } else {
                std::hint::spin_loop();
            }
        }
    }

    /// Start the timer worker.
    ///
    /// # Errors
    /// Returns the OS error if the worker cannot be started.
    pub fn new() -> io::Result<Self> {
        let shared = Arc::new(Shared::default());
        let worker_state = Arc::clone(&shared);
        let worker = std::thread::Builder::new()
            .name("deq-input-deadlines".into())
            .spawn(move || run(&worker_state))?;
        Ok(Self {
            shared,
            worker: Some(worker),
        })
    }

    /// Wait until `deadline`, removing the queued wakeup if this future is dropped.
    ///
    /// # Errors
    /// Returns an error if the timer worker stops or its sequence is exhausted.
    ///
    /// # Panics
    /// Panics if the timer state mutex was poisoned.
    pub async fn wait_until(&self, deadline: Instant) -> io::Result<()> {
        let now = Instant::now();
        let remaining = deadline.saturating_duration_since(now);
        if remaining.is_zero() {
            return Ok(());
        }
        let wake_at = deadline - remaining.min(Self::BLOCKING_PRECISION_WINDOW);
        let (sender, receiver) = oneshot::channel();
        let key = {
            let mut state = self.shared.state.lock().unwrap();
            state.sequence = state
                .sequence
                .checked_add(1)
                .ok_or_else(|| io::Error::other("deadline sequence exhausted"))?;
            let key = (wake_at, state.sequence);
            let earliest = state.pending.first_key_value().is_none_or(|(first, _)| key < *first);
            state.pending.insert(key, sender);
            if earliest {
                self.shared.changed.notify_one();
            }
            key
        };
        let _pending = Pending {
            shared: Arc::clone(&self.shared),
            key,
        };
        receiver
            .await
            .map_err(|_| io::Error::other("input deadline worker stopped"))?;
        while Instant::now() < deadline {
            tokio::task::yield_now().await;
        }
        Ok(())
    }
}

struct Pending {
    shared: Arc<Shared>,
    key: (Instant, u64),
}

impl Drop for Pending {
    fn drop(&mut self) {
        let mut state = self.shared.state.lock().unwrap();
        if state.pending.remove(&self.key).is_some() {
            self.shared.changed.notify_one();
        }
    }
}

fn run(shared: &Shared) {
    let mut state = shared.state.lock().unwrap();
    loop {
        if state.stopping {
            return;
        }
        let Some((&(deadline, _), _)) = state.pending.first_key_value() else {
            state = shared.changed.wait(state).unwrap();
            continue;
        };
        let remaining = deadline.saturating_duration_since(Instant::now());
        if !remaining.is_zero() {
            state = shared.changed.wait_timeout(state, remaining).unwrap().0;
            continue;
        }
        let now = Instant::now();
        let mut ready = Vec::new();
        while state
            .pending
            .first_key_value()
            .is_some_and(|((deadline, _), _)| *deadline <= now)
        {
            ready.push(state.pending.pop_first().unwrap());
        }
        drop(state);
        for ((deadline, sequence), sender) in ready {
            let mut trace = crate::decoder::worker_trace::WorkerTrace::root(
                "deadline_wakeup",
                crate::misc::fastrace::system_diagnostics_enabled(),
            );
            trace.property("sequence", || sequence.to_string());
            trace.property("late_ns", || {
                Instant::now().saturating_duration_since(deadline).as_nanos().to_string()
            });
            let _ = sender.send(());
            trace.event("sent");
        }
        state = shared.state.lock().unwrap();
    }
}

impl Drop for DeadlineTimer {
    fn drop(&mut self) {
        {
            let mut state = self.shared.state.lock().unwrap();
            state.stopping = true;
            state.pending.clear();
            self.shared.changed.notify_one();
        }
        if self.worker.take().unwrap().join().is_err() {
            eprintln!("input deadline worker panicked");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocking_emitter_never_submits_before_its_deadline() {
        let deadline = Instant::now() + Duration::from_millis(2);
        DeadlineTimer::wait_until_blocking(deadline);
        assert!(Instant::now() >= deadline);
        DeadlineTimer::wait_until_blocking(deadline);
    }

    #[tokio::test]
    async fn earlier_deadlines_interrupt_waiting_without_early_delivery() {
        let timer = DeadlineTimer::new().unwrap();
        let now = Instant::now();
        let late = now + Duration::from_millis(50);
        let early = now + Duration::from_millis(5);
        let later = timer.wait_until(late);
        tokio::pin!(later);
        assert!(futures_util::poll!(&mut later).is_pending());
        timer.wait_until(early).await.unwrap();
        assert!(Instant::now() >= early);
        later.await.unwrap();
        assert!(Instant::now() >= late);
    }

    #[tokio::test]
    async fn cancelled_deadlines_release_all_queued_state() {
        let timer = DeadlineTimer::new().unwrap();
        let deadline = Instant::now() + Duration::from_hours(1);
        let mut waits = Vec::new();
        for _ in 0..1000 {
            let mut wait = Box::pin(timer.wait_until(deadline));
            assert!(futures_util::poll!(&mut wait).is_pending());
            waits.push(wait);
        }
        assert_eq!(timer.shared.state.lock().unwrap().pending.len(), 1000);
        drop(waits);
        assert!(timer.shared.state.lock().unwrap().pending.is_empty());
        timer.wait_until(Instant::now()).await.unwrap();
        assert!(timer.shared.state.lock().unwrap().pending.is_empty());
    }

    #[tokio::test]
    async fn async_wakeup_reserves_the_precision_window() {
        let timer = DeadlineTimer::new().unwrap();
        let deadline = Instant::now() + Duration::from_hours(1);
        let mut wait = Box::pin(timer.wait_until(deadline));
        assert!(futures_util::poll!(&mut wait).is_pending());
        let wake_at = timer.shared.state.lock().unwrap().pending.first_key_value().unwrap().0.0;
        assert_eq!(wake_at, deadline - DeadlineTimer::BLOCKING_PRECISION_WINDOW);
        drop(wait);
        assert!(timer.shared.state.lock().unwrap().pending.is_empty());
    }

    #[tokio::test]
    async fn precision_window_accepts_cancellation_and_earlier_deadlines() {
        let timer = DeadlineTimer::new().unwrap();
        let now = Instant::now();
        let mut cancelled = Box::pin(timer.wait_until(now + Duration::from_micros(400)));
        let _ = futures_util::poll!(&mut cancelled);
        drop(cancelled);
        assert!(timer.shared.state.lock().unwrap().pending.is_empty());

        let later_deadline = Instant::now() + Duration::from_micros(400);
        let earlier_deadline = later_deadline - Duration::from_micros(200);
        let mut later = Box::pin(timer.wait_until(later_deadline));
        let _ = futures_util::poll!(&mut later);
        timer.wait_until(earlier_deadline).await.unwrap();
        assert!(Instant::now() >= earlier_deadline);
        later.await.unwrap();
        assert!(Instant::now() >= later_deadline);
        assert!(timer.shared.state.lock().unwrap().pending.is_empty());
    }
}
