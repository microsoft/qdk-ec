//! Unit tests for the `TaskCounter`/`TaskGuard` barrier primitive.

use deq_runtime::misc::sync::TaskCounter;
use futures_util::FutureExt;
use std::sync::Arc;

#[tokio::test]
async fn test_task_counter_zero_immediately() {
    let counter = TaskCounter::new();
    assert!(counter.wait_for_zero().now_or_never().is_some());
}

#[tokio::test]
async fn test_task_counter_waits_for_guard() {
    let counter = TaskCounter::new();
    let guard = counter.guard();

    assert!(counter.wait_for_zero().now_or_never().is_none());

    drop(guard);
    assert!(counter.wait_for_zero().now_or_never().is_some());
}

#[tokio::test]
async fn test_task_counter_multiple_guards() {
    let counter = TaskCounter::new();
    let g1 = counter.guard();
    let g2 = counter.guard();
    let g3 = counter.guard();

    drop(g1);
    assert!(counter.wait_for_zero().now_or_never().is_none());

    drop(g2);
    assert!(counter.wait_for_zero().now_or_never().is_none());

    drop(g3);
    assert!(counter.wait_for_zero().now_or_never().is_some());
}

#[tokio::test]
async fn test_task_counter_guard_drop_on_panic() {
    let counter = TaskCounter::new();
    let _guard = counter.guard();

    // Spawn a task that panics while holding a guard
    let counter2 = Arc::clone(&counter);
    let handle = tokio::spawn(async move {
        let _guard = counter2.guard();
        panic!("intentional panic");
    });

    // The panicking task's guard should still decrement
    let _ = handle.await; // JoinError (panic)

    // Drop the outer guard
    drop(_guard);

    assert!(counter.wait_for_zero().now_or_never().is_some());
}

#[tokio::test]
async fn test_task_counter_pause_blocks_new_operations_and_reopens_on_drop() {
    let counter = TaskCounter::new();
    let active = counter.try_guard().expect("operation should be admitted");
    let pause = counter.try_pause().expect("first reset should pause admission");

    assert!(counter.try_guard().is_none());
    assert!(counter.try_pause().is_none());

    assert!(counter.wait_for_zero().now_or_never().is_none());
    drop(active);
    assert!(counter.wait_for_zero().now_or_never().is_some());

    drop(pause);
    assert!(counter.try_guard().is_some());
}
