use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tokio::sync::{Notify, watch};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

pub fn get_or_receiver<T: Send + Sync + Clone + 'static>(
    sender: &watch::Sender<Option<T>>,
    token: CancellationToken,
) -> Result<T, JoinHandle<Option<T>>> {
    get_or_future(sender, token).map_err(tokio::spawn)
}

#[allow(
    clippy::result_large_err,
    reason = "The alternative is inline wait state, not an error; avoid a separate allocation."
)]
pub(crate) fn get_or_future<T: Send + Sync + Clone + 'static>(
    sender: &watch::Sender<Option<T>>,
    token: CancellationToken,
) -> Result<T, impl Future<Output = Option<T>> + Send + 'static + use<T>> {
    if let Some(value) = sender.borrow().as_ref() {
        Ok(value.clone())
    } else {
        let mut rx = sender.subscribe();
        Err(async move {
            tokio::select! {
                result = rx.wait_for(|v| v.is_some()) => {
                    result.ok().and_then(|v| v.clone())
                }
                _ = token.cancelled() => None,
            }
        })
    }
}

pub async fn get_value<T: Send + Sync + Clone + 'static>(
    sender: &watch::Sender<Option<T>>,
    token: CancellationToken,
) -> Option<T> {
    match get_or_receiver(sender, token) {
        Ok(value) => Some(value),
        Err(handle) => handle.await.unwrap_or(None),
    }
}

/// Returns `Ok(())` if the value already exists, or spawns a task that waits
/// for it (or cancellation). The spawned task returns `true` if the value
/// arrived and `false` if cancelled.
pub fn check_or_receiver<T: Send + Sync + Clone + 'static>(
    sender: &watch::Sender<Option<T>>,
    token: CancellationToken,
) -> Result<(), JoinHandle<bool>> {
    check_or_future(sender, token).map_err(tokio::spawn)
}

#[allow(
    clippy::result_large_err,
    reason = "The alternative is inline wait state, not an error; avoid a separate allocation."
)]
pub(crate) fn check_or_future<T: Send + Sync + Clone + 'static>(
    sender: &watch::Sender<Option<T>>,
    token: CancellationToken,
) -> Result<(), impl Future<Output = bool> + Send + 'static + use<T>> {
    if sender.borrow().as_ref().is_some() {
        Ok(())
    } else {
        let mut rx = sender.subscribe();
        Err(async move {
            tokio::select! {
                result = rx.wait_for(|v| v.is_some()) => result.is_ok(),
                _ = token.cancelled() => false,
            }
        })
    }
}

// ============================================================================
// TaskCounter / TaskGuard — track active spawned tasks for barrier-based reset
// ============================================================================

/// Tracks the number of active background tasks. `reset()` cancels the token
/// then calls `wait_for_zero()` to ensure all tasks have exited before clearing
/// shared state.
pub struct TaskCounter {
    count: AtomicUsize,
    accepting: AtomicBool,
    notify: Notify,
}

impl TaskCounter {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            count: AtomicUsize::new(0),
            accepting: AtomicBool::new(true),
            notify: Notify::new(),
        })
    }

    /// Wait until the active task count reaches zero.
    pub async fn wait_for_zero(self: &Arc<Self>) {
        loop {
            let notified = self.notify.notified();
            // Acquire pairs with the Release in TaskGuard::drop, ensuring
            // all writes made by finished tasks are visible after we return.
            if self.count.load(Ordering::Acquire) == 0 {
                return;
            }
            // The `Notified` future is registered before we re-check, so if
            // a guard is dropped between `load` and here, the notification
            // is captured by the already-registered waiter — no lost wakeup.
            notified.await;
        }
    }

    /// Create a guard that increments the counter now and decrements on drop.
    pub fn guard(self: &Arc<Self>) -> TaskGuard {
        self.count.fetch_add(1, Ordering::Relaxed);
        TaskGuard {
            counter: Arc::clone(self),
        }
    }

    /// Admit a new top-level operation unless reset has paused admission.
    /// A second acceptance check closes the race with [`Self::try_pause`].
    pub fn try_guard(self: &Arc<Self>) -> Option<TaskGuard> {
        if !self.accepting.load(Ordering::Acquire) {
            return None;
        }
        let guard = self.guard();
        if self.accepting.load(Ordering::Acquire) {
            Some(guard)
        } else {
            drop(guard);
            None
        }
    }

    /// Stop admitting top-level operations until the returned guard is dropped.
    /// Returns `None` when another reset already owns the pause.
    pub fn try_pause(self: &Arc<Self>) -> Option<TaskCounterPause> {
        self.accepting
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|_| TaskCounterPause {
                counter: Arc::clone(self),
            })
    }
}

/// RAII guard that decrements the [`TaskCounter`] when dropped.
/// Move this into each spawned task to track its lifetime.
pub struct TaskGuard {
    counter: Arc<TaskCounter>,
}

impl Drop for TaskGuard {
    fn drop(&mut self) {
        // Release ensures all writes made by this task are visible to the
        // thread that later observes count == 0 with an Acquire load.
        if self.counter.count.fetch_sub(1, Ordering::Release) == 1 {
            self.counter.notify.notify_waiters();
        }
    }
}

/// RAII admission pause used by reset. Dropping it reopens the coordinator even
/// when reset returns early with an error.
pub struct TaskCounterPause {
    counter: Arc<TaskCounter>,
}

impl Drop for TaskCounterPause {
    fn drop(&mut self) {
        self.counter.accepting.store(true, Ordering::Release);
    }
}

#[cfg(test)]
mod watch_future_tests {
    use super::*;
    use std::task::Poll;

    #[tokio::test]
    async fn ready_watch_values_do_not_require_another_task_turn() {
        let (sender, _) = watch::channel(None::<u64>);
        let Err(waiting) = get_or_future(&sender, CancellationToken::new()) else {
            panic!("value is not ready");
        };
        sender.send_replace(Some(7));
        tokio::pin!(waiting);
        assert!(matches!(futures_util::poll!(&mut waiting), Poll::Ready(Some(7))));
        assert!(matches!(get_or_future(&sender, CancellationToken::new()), Ok(7)));
    }

    #[tokio::test]
    async fn dropping_an_unpolled_wait_releases_its_subscription() {
        let (sender, _) = watch::channel(None::<()>);
        let Err(waiting) = check_or_future(&sender, CancellationToken::new()) else {
            panic!("value is not ready");
        };
        assert_eq!(sender.receiver_count(), 1);
        drop(waiting);
        assert_eq!(sender.receiver_count(), 0);
    }

    #[tokio::test]
    async fn inline_watch_waits_preserve_cancellation_and_closed_channel_results() {
        let (sender, _) = watch::channel(None::<()>);
        let token = CancellationToken::new();
        let Err(waiting) = check_or_future(&sender, token.clone()) else {
            panic!("value is not ready");
        };
        token.cancel();
        assert!(!waiting.await);
        let Err(waiting) = get_or_future(&sender, CancellationToken::new()) else {
            panic!("value is not ready");
        };
        drop(sender);
        assert!(waiting.await.is_none());
    }
}
