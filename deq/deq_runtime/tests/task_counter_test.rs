//! Unit tests for the `TaskCounter`/`TaskGuard` barrier primitive.

use deq_runtime::misc::sync::TaskCounter;
use std::sync::Arc;
use std::time::Duration;

#[tokio::test]
async fn test_task_counter_zero_immediately() {
    let counter = TaskCounter::new();
    // No guards — should return immediately
    tokio::time::timeout(Duration::from_millis(100), counter.wait_for_zero())
        .await
        .expect("wait_for_zero should return immediately when count is 0");
}

#[tokio::test]
async fn test_task_counter_waits_for_guard() {
    let counter = TaskCounter::new();
    let guard = counter.guard();

    // wait_for_zero should NOT complete while guard is alive
    let counter2 = Arc::clone(&counter);
    let handle = tokio::spawn(async move {
        counter2.wait_for_zero().await;
    });

    // Give it a moment to start waiting
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(!handle.is_finished(), "wait_for_zero should still be waiting");

    // Drop the guard — wait_for_zero should complete
    drop(guard);
    tokio::time::timeout(Duration::from_millis(100), handle)
        .await
        .expect("should complete after guard drop")
        .expect("task should not panic");
}

#[tokio::test]
async fn test_task_counter_multiple_guards() {
    let counter = TaskCounter::new();
    let g1 = counter.guard();
    let g2 = counter.guard();
    let g3 = counter.guard();

    let counter2 = Arc::clone(&counter);
    let handle = tokio::spawn(async move {
        counter2.wait_for_zero().await;
    });

    drop(g1);
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(!handle.is_finished(), "should still wait with 2 guards");

    drop(g2);
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(!handle.is_finished(), "should still wait with 1 guard");

    drop(g3);
    tokio::time::timeout(Duration::from_millis(100), handle)
        .await
        .expect("should complete after all guards dropped")
        .expect("task should not panic");
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

    // Now wait_for_zero should complete
    tokio::time::timeout(Duration::from_millis(100), counter.wait_for_zero())
        .await
        .expect("wait_for_zero should complete after panic + guard drop");
}
