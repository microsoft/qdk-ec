//! Bounded CPU work on one frame-state owner, independent of caller scheduling.

use crate::misc::pauli_frame_tracker::PauliFrameTracker;
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;
use tonic::Status;

type FrameWork = Box<dyn FnOnce(&mut PauliFrameTracker) + Send>;

enum Command {
    Run(FrameWork),
    Drain(oneshot::Sender<()>),
}

pub(super) struct FrameExecutor {
    sender: mpsc::Sender<Command>,
    trace_sequence: AtomicU64,
}

impl FrameExecutor {
    pub(super) fn new(tracker: Arc<Mutex<PauliFrameTracker>>, capacity: usize) -> io::Result<Self> {
        let (sender, mut receiver) = mpsc::channel(capacity);
        std::thread::Builder::new().name("deq-frame-owner".into()).spawn(move || {
            while let Some(command) = receiver.blocking_recv() {
                match command {
                    Command::Run(work) => {
                        if tracker.is_poisoned() {
                            drop(work);
                            continue;
                        }
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let mut tracker = tracker.lock().unwrap();
                            work(&mut tracker);
                        }));
                        if result.is_err() {
                            eprintln!("Pauli frame worker panicked; coordinator reset is required");
                        }
                    }
                    Command::Drain(completed) => {
                        let _ = completed.send(());
                    }
                }
            }
        })?;
        Ok(Self {
            sender,
            trace_sequence: AtomicU64::new(0),
        })
    }

    pub(super) async fn run<T: Send + 'static>(
        &self,
        token: &CancellationToken,
        work: impl FnOnce(&mut PauliFrameTracker) -> T + Send + 'static,
    ) -> Result<T, Status> {
        if crate::misc::fastrace::system_diagnostics_enabled() {
            let sequence = self.trace_sequence.fetch_add(1, Ordering::Relaxed);
            if sequence.is_multiple_of(64) {
                // Keep optional diagnostic state out of the ordinary caller future.
                return Box::pin(self.run_traced(token, work, sequence)).await;
            }
        }
        self.run_inner(token, work).await
    }

    async fn run_traced<T: Send + 'static>(
        &self,
        token: &CancellationToken,
        work: impl FnOnce(&mut PauliFrameTracker) -> T + Send + 'static,
        sequence: u64,
    ) -> Result<T, Status> {
        let mut trace = crate::decoder::worker_trace::WorkerTrace::root("frame_operation", true);
        trace.property("sequence", || sequence.to_string());
        trace.property("sample_rate", || "64".into());
        let submitted = std::time::Instant::now();
        let (result, mut trace, started, finished) = self
            .run_inner(token, move |tracker| {
                let started = std::time::Instant::now();
                let result = work(tracker);
                let finished = std::time::Instant::now();
                (result, trace, started, finished)
            })
            .await?;
        let resumed = std::time::Instant::now();
        trace.event_at("submission_started", submitted);
        trace.event_at("worker_started", started);
        trace.event_at("worker_finished", finished);
        trace.event_at("caller_resumed", resumed);
        Ok(result)
    }

    async fn run_inner<T: Send + 'static>(
        &self,
        token: &CancellationToken,
        work: impl FnOnce(&mut PauliFrameTracker) -> T + Send + 'static,
    ) -> Result<T, Status> {
        let (completed, result) = oneshot::channel();
        let command = Command::Run(Box::new(move |tracker| {
            let result = work(tracker);
            let _ = completed.send(result);
        }));
        let cancelled = token.cancelled();
        tokio::pin!(cancelled);
        tokio::select! {
            biased;
            () = &mut cancelled => return Err(Status::cancelled("frame operation cancelled")),
            sent = self.sender.send(command) => {
                sent.map_err(|_| Status::unavailable("Pauli frame worker stopped"))?;
            }
        }
        tokio::select! {
            biased;
            () = &mut cancelled => Err(Status::cancelled("frame operation cancelled")),
            result = result => result.map_err(|_| Status::internal("frame operation failed; reset the coordinator")),
        }
    }

    pub(super) async fn drain(&self) -> Result<(), Status> {
        let (completed, drained) = oneshot::channel();
        self.sender
            .send(Command::Drain(completed))
            .await
            .map_err(|_| Status::unavailable("Pauli frame worker stopped before drain"))?;
        drained
            .await
            .map_err(|_| Status::internal("Pauli frame worker failed to drain"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::FutureExt as _;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[tokio::test]
    async fn cancellation_does_not_remove_running_work_from_the_drain_boundary() {
        let tracker = Arc::new(Mutex::new(PauliFrameTracker::new()));
        let executor = Arc::new(FrameExecutor::new(tracker, 2).unwrap());
        let token = CancellationToken::new();
        let finished = Arc::new(AtomicBool::new(false));
        let (started, ready) = oneshot::channel();
        let (release, blocked) = std::sync::mpsc::sync_channel(0);
        let request = tokio::spawn({
            let executor = Arc::clone(&executor);
            let token = token.clone();
            let finished = Arc::clone(&finished);
            async move {
                executor
                    .run(&token, move |_| {
                        started.send(()).unwrap();
                        blocked.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
                        finished.store(true, Ordering::Release);
                    })
                    .await
            }
        });
        ready.await.unwrap();
        token.cancel();
        assert_eq!(request.await.unwrap().unwrap_err().code(), tonic::Code::Cancelled);
        let mut drain = Box::pin(executor.drain());
        assert!(drain.as_mut().now_or_never().is_none());
        assert!(!finished.load(Ordering::Acquire));
        release.send(()).unwrap();
        drain.await.unwrap();
        assert!(finished.load(Ordering::Acquire));
    }
}
