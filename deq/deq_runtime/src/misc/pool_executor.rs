//! Async-task scheduling on the decoder's existing Rayon pool, with optional caller participation.
//!
//! `async_task` owns future polling and cancellation; Rayon owns threads and the shared work queue.
//! This adapter keeps one ready continuation local to avoid a handoff, then yields after a bounded
//! number of polls. Pending futures release their worker rather than blocking on other pool jobs.
//! Tokio still drives I/O and timers; each runnable enters the runtime that spawned it.
//!
//! Thread-local state is scoped to a poll and restored on return or unwind. Never hold a
//! `RefCell` borrow while polling a future: waking another task can reenter the scheduler.

use std::cell::{Cell, RefCell};
use std::future::Future;
use std::io;
use std::sync::Arc;
use std::task::{Context, Poll};

/// Cooperative scheduling heuristic, not a correctness limit; overflow returns to Rayon's queue.
const MAX_LOCAL_POLLS: usize = 3;

/// Wakes a participating caller; park/unpark's token handles wakes arriving before the caller parks.
struct Caller {
    thread: std::thread::Thread,
}

impl futures_util::task::ArcWake for Caller {
    fn wake_by_ref(caller: &Arc<Self>) {
        caller.thread.unpark();
    }
}

thread_local! {
    static CALLER: RefCell<Option<Arc<Caller>>> = const { RefCell::new(None) };
    static INLINE_CPU: Cell<bool> = const { Cell::new(false) };
}

/// Allows synchronous solver work only inside a computation task, not the input-delivery future.
struct InlineScope(bool);

impl InlineScope {
    fn enter(enabled: bool) -> Self {
        Self(INLINE_CPU.with(|current| current.replace(enabled)))
    }
}

impl Drop for InlineScope {
    fn drop(&mut self) {
        INLINE_CPU.with(|current| current.set(self.0));
    }
}

/// Use the caller's immediate slot when available; a one-thread pool must also drive its own overflow.
pub(crate) fn can_submit_inline(pool: &rayon::ThreadPool) -> bool {
    pool.current_thread_index().is_some()
        && CALLER.with(|caller| caller.borrow().is_some())
        && LOCAL_POLL.with(|local| {
            let local = local.borrow();
            local.active && (local.next.is_none() || pool.current_num_threads() == 1)
        })
}

pub(crate) fn inline_cpu_allowed() -> bool {
    INLINE_CPU.with(Cell::get)
}

pub(crate) fn execute_cpu(work: impl FnOnce()) {
    let _scope = InlineScope::enter(true);
    work();
}

/// Propagates the caller wakeup target across worker polls without leaking it into unrelated tasks.
struct CallerScope(Option<Arc<Caller>>);

impl CallerScope {
    fn enter(caller: Option<Arc<Caller>>) -> Self {
        Self(CALLER.with(|current| current.replace(caller)))
    }
}

impl Drop for CallerScope {
    fn drop(&mut self) {
        CALLER.with(|current| current.replace(self.0.take()));
    }
}

struct Scheduled {
    runnable: async_task::Runnable,
    runtime: tokio::runtime::Handle,
    caller: Option<Arc<Caller>>,
}

impl Scheduled {
    fn run(self) {
        let _runtime = self.runtime.enter();
        let _caller = CallerScope::enter(self.caller);
        let _cpu = InlineScope::enter(true);
        self.runnable.run();
    }
}

/// A single ready continuation, not a second unbounded queue.
#[derive(Default)]
struct LocalPoll {
    active: bool,
    next: Option<Scheduled>,
}

thread_local! {
    static LOCAL_POLL: RefCell<LocalPoll> = const {
        RefCell::new(LocalPoll { active: false, next: None })
    };
}

struct PollScope {
    previous: LocalPoll,
    pool: Arc<rayon::ThreadPool>,
}

impl PollScope {
    fn enter(pool: Arc<rayon::ThreadPool>) -> Self {
        let previous = LOCAL_POLL.with(|local| {
            local.replace(LocalPoll {
                active: true,
                next: None,
            })
        });
        Self { previous, pool }
    }
}

impl Drop for PollScope {
    fn drop(&mut self) {
        let remaining = LOCAL_POLL
            .with(|local| local.replace(std::mem::take(&mut self.previous)))
            .next;
        if let Some(runnable) = remaining {
            enqueue(&self.pool, runnable);
        }
    }
}

fn enqueue(pool: &Arc<rayon::ThreadPool>, mut scheduled: Scheduled) {
    let owner = Arc::clone(pool);
    pool.spawn_fifo(move || {
        let _scope = PollScope::enter(Arc::clone(&owner));
        for _ in 0..MAX_LOCAL_POLLS {
            scheduled.run();
            let Some(next) = LOCAL_POLL.with(|local| local.borrow_mut().next.take()) else {
                return;
            };
            scheduled = next;
        }
        enqueue(&owner, scheduled);
    });
}

/// Publish the local continuation before lengthy CPU work so another worker can pick it up.
pub(crate) fn flush_local(pool: &Arc<rayon::ThreadPool>) {
    if pool.current_thread_index().is_some()
        && let Some(scheduled) = LOCAL_POLL.with(|local| local.borrow_mut().next.take())
    {
        enqueue(pool, scheduled);
    }
}

/// Drive a future on a caller that participates in the decoder's worker budget.
///
/// Construct the pool using `ThreadPoolBuilder::use_current_thread`. Ready
/// continuations can run on the input thread; the other pool workers retain
/// parallel capacity. Pending work parks the caller instead of spinning.
///
/// # Errors
/// Rejects a caller outside this pool or a single-threaded Tokio runtime,
/// whose I/O driver would not otherwise make progress.
pub fn block_on<F: Future>(
    pool: &Arc<rayon::ThreadPool>,
    runtime: &tokio::runtime::Handle,
    future: F,
) -> io::Result<F::Output> {
    if pool.current_thread_index().is_none() || runtime.runtime_flavor() != tokio::runtime::RuntimeFlavor::MultiThread {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "caller execution requires membership in this pool and a multi-threaded Tokio runtime",
        ));
    }
    let caller = Arc::new(Caller {
        thread: std::thread::current(),
    });
    let waker = futures_util::task::waker_ref(&caller);
    let mut context = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    let _runtime = runtime.enter();
    let _caller = CallerScope::enter(Some(Arc::clone(&caller)));
    loop {
        let result = {
            let _scope = PollScope::enter(Arc::clone(pool));
            let result = {
                let _cpu = InlineScope::enter(false);
                future.as_mut().poll(&mut context)
            };
            for _ in 0..MAX_LOCAL_POLLS {
                let Some(scheduled) = LOCAL_POLL.with(|local| local.borrow_mut().next.take()) else {
                    break;
                };
                scheduled.run();
            }
            result
        };
        if let Poll::Ready(output) = result {
            if pool.current_num_threads() == 1 {
                while matches!(pool.yield_now(), Some(rayon::Yield::Executed)) {}
            }
            return Ok(output);
        }
        if !matches!(pool.yield_now(), Some(rayon::Yield::Executed)) {
            std::thread::park();
        }
    }
}

/// Poll asynchronous CPU work on its existing bounded worker pool.
///
/// Pending futures release the worker; wakes enqueue another poll, never block
/// a worker waiting for another job in the same pool. Dropping the returned task
/// cancels its future, including the task's admitted-operation guards.
pub(crate) fn spawn<F>(
    pool: Arc<rayon::ThreadPool>,
    runtime: tokio::runtime::Handle,
    future: F,
) -> async_task::Task<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    let caller = CALLER.with(|current| current.borrow().clone());
    let schedule = move |runnable: async_task::Runnable| {
        let scheduled = Scheduled {
            runnable,
            runtime: runtime.clone(),
            caller: caller.clone(),
        };
        let queued = if pool.current_thread_index().is_some() {
            LOCAL_POLL.with(|local| {
                let mut local = local.borrow_mut();
                if local.active {
                    // Keep the newest wake local; publish any displaced continuation to Rayon.
                    local.next.replace(scheduled)
                } else {
                    Some(scheduled)
                }
            })
        } else {
            Some(scheduled)
        };
        if let Some(scheduled) = queued {
            enqueue(&pool, scheduled);
        }
        if let Some(caller) = &caller {
            caller.thread.unpark();
        }
    };
    let (runnable, task) = async_task::spawn(future, schedule);
    runnable.schedule();
    task
}

#[cfg(test)]
#[path = "../../tests/unit/pool_executor_test.rs"]
mod tests;
