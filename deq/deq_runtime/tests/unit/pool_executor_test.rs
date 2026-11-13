use super::*;
use std::time::Duration;
use tokio::sync::oneshot;

#[tokio::test]
async fn local_poll_budget_yields_to_already_queued_work() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap());
    let polls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::new(AtomicUsize::new(0));
    let queue = Arc::clone(&pool);
    let observed_by_task = Arc::clone(&observed);
    let task = spawn(
        pool,
        tokio::runtime::Handle::current(),
        std::future::poll_fn(move |context| {
            let count = polls.fetch_add(1, Ordering::SeqCst) + 1;
            if count == 1 {
                let polls = Arc::clone(&polls);
                let observed = Arc::clone(&observed_by_task);
                queue.spawn_fifo(move || observed.store(polls.load(Ordering::SeqCst), Ordering::SeqCst));
            }
            if count < MAX_LOCAL_POLLS + 2 {
                context.waker().wake_by_ref();
                Poll::Pending
            } else {
                Poll::Ready(())
            }
        }),
    );
    tokio::time::timeout(Duration::from_secs(2), task).await.unwrap();
    assert_eq!(observed.load(Ordering::SeqCst), MAX_LOCAL_POLLS);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_full_caller_slot_leaves_other_requests_on_the_async_executor() {
    let runtime = tokio::runtime::Handle::current();
    let (finished, result) = oneshot::channel();
    let thread = std::thread::spawn(move || {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(2)
                .use_current_thread()
                .build()
                .unwrap(),
        );
        block_on(&pool, &runtime, async {
            assert!(!inline_cpu_allowed());
            assert!(can_submit_inline(&pool));
            let child = spawn(Arc::clone(&pool), runtime.clone(), async { inline_cpu_allowed() });
            assert!(!can_submit_inline(&pool));
            assert!(child.await);
        })
        .unwrap();
        finished.send(()).unwrap();
    });
    tokio::time::timeout(Duration::from_secs(2), result).await.unwrap().unwrap();
    thread.join().unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_single_caller_worker_drives_dependencies_and_borrowed_futures() {
    let runtime = tokio::runtime::Handle::current();
    let (finished, result) = oneshot::channel();
    let thread = std::thread::spawn(move || {
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .use_current_thread()
                .build()
                .unwrap(),
        );
        let caller = std::thread::current().id();
        let mut value = 0;
        block_on(&pool, &runtime, async {
            let dependency = spawn(Arc::clone(&pool), runtime.clone(), async {
                tokio::time::sleep(Duration::from_millis(5)).await;
                std::thread::current().id()
            });
            assert_eq!(dependency.await, caller);
            value = 7;
        })
        .unwrap();
        block_on(&pool, &runtime, async { value += 1 }).unwrap();
        finished.send(value).unwrap();
    });
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), result).await.unwrap().unwrap(),
        8
    );
    thread.join().unwrap();
}

#[tokio::test]
async fn local_continuations_keep_their_own_runtime_context() {
    let other = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap();
    let other_handle = other.handle().clone();
    let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap());
    let parent = spawn(Arc::clone(&pool), tokio::runtime::Handle::current(), async move {
        let child = spawn(pool, other_handle, async {
            tokio::runtime::Handle::current().runtime_flavor()
        });
        assert_eq!(child.await, tokio::runtime::RuntimeFlavor::MultiThread);
        assert_eq!(
            tokio::runtime::Handle::current().runtime_flavor(),
            tokio::runtime::RuntimeFlavor::CurrentThread
        );
    });
    tokio::time::timeout(Duration::from_secs(2), parent).await.unwrap();
    other.shutdown_background();
}

#[tokio::test]
async fn a_ready_continuation_stays_on_its_current_worker() {
    let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap());
    let runtime = tokio::runtime::Handle::current();
    let parent = spawn(Arc::clone(&pool), runtime.clone(), async move {
        let original = std::thread::current().id();
        let child = spawn(pool, runtime, async { std::thread::current().id() });
        assert_eq!(child.await, original);
        assert_eq!(std::thread::current().id(), original);
    });
    tokio::time::timeout(Duration::from_secs(2), parent).await.unwrap();
}

#[tokio::test]
async fn pending_work_releases_a_single_worker_for_its_dependency() {
    let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap());
    let runtime = tokio::runtime::Handle::current();
    let (sender, receiver) = oneshot::channel();
    let waiting = spawn(Arc::clone(&pool), runtime.clone(), async move { receiver.await.unwrap() });
    let supplying = spawn(pool, runtime, async move { sender.send(7).unwrap() });
    tokio::time::timeout(Duration::from_secs(2), async {
        supplying.await;
        assert_eq!(waiting.await, 7);
    })
    .await
    .unwrap();
}

#[test]
fn nested_scopes_restore_thread_state_after_unwinding() {
    let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap());
    let caller = Arc::new(Caller {
        thread: std::thread::current(),
    });
    let _caller = CallerScope::enter(Some(Arc::clone(&caller)));
    let _poll = PollScope::enter(Arc::clone(&pool));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _poll = PollScope::enter(Arc::clone(&pool));
        let _caller = CallerScope::enter(None);
        execute_cpu(|| {
            assert!(inline_cpu_allowed());
            panic!("exercise scope cleanup");
        });
    }));
    assert!(result.is_err());
    assert!(!inline_cpu_allowed());
    assert!(CALLER.with(|current| Arc::ptr_eq(current.borrow().as_ref().unwrap(), &caller)));
    assert!(LOCAL_POLL.with(|local| local.borrow().active));
}

#[tokio::test]
async fn leaving_a_poll_scope_publishes_its_unpolled_continuation() {
    let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap());
    let scope = PollScope::enter(Arc::clone(&pool));
    let (runnable, task) = async_task::spawn(async { 42 }, |_| panic!("already-ready task must not be rescheduled"));
    LOCAL_POLL.with(|local| {
        local.borrow_mut().next = Some(Scheduled {
            runnable,
            runtime: tokio::runtime::Handle::current(),
            caller: None,
        });
    });
    drop(scope);
    assert!(!LOCAL_POLL.with(|local| local.borrow().active));
    assert_eq!(tokio::time::timeout(Duration::from_secs(2), task).await.unwrap(), 42);
}
