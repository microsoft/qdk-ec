use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::{Notify, watch};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

pub fn get_or_receiver<T: Send + Sync + Clone + 'static>(
    sender: &watch::Sender<Option<T>>,
    token: CancellationToken,
) -> Result<T, JoinHandle<Option<T>>> {
    if let Some(value) = sender.borrow().as_ref() {
        Ok(value.clone())
    } else {
        let mut rx = sender.subscribe();
        Err(tokio::spawn(async move {
            tokio::select! {
                result = rx.wait_for(|v| v.is_some()) => {
                    result.ok().and_then(|v| v.clone())
                }
                _ = token.cancelled() => None,
            }
        }))
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
    if sender.borrow().as_ref().is_some() {
        Ok(())
    } else {
        let mut rx = sender.subscribe();
        Err(tokio::spawn(async move {
            tokio::select! {
                result = rx.wait_for(|v| v.is_some()) => result.is_ok(),
                _ = token.cancelled() => false,
            }
        }))
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
    notify: Notify,
}

impl TaskCounter {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            count: AtomicUsize::new(0),
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
