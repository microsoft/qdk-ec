use super::*;

#[test]
fn inactive_worker_trace_does_not_evaluate_properties() {
    let trace = WorkerTrace::new("inactive");
    #[cfg(feature = "cli")]
    assert!(trace.active.is_none());
    trace.property("unused", || panic!("inactive trace must not format properties"));
    trace.event("worker_started");
    trace.solved(true);
}

#[cfg(feature = "cli")]
#[test]
fn unsampled_parent_does_not_start_worker_timing() {
    let mut context = SpanContext::random();
    context.sampled = false;
    let span = Span::root("unsampled", context);
    let _parent = span.set_local_parent();
    let trace = WorkerTrace::new("inactive");
    assert!(trace.active.is_none());
    trace.property("unused", || panic!("unsampled trace must not format properties"));
}

#[cfg(feature = "cli")]
#[test]
fn sampled_parent_without_reporter_does_not_start_worker_timing() {
    // Use a fresh process so another test cannot install the global reporter.
    const CHILD: &str = "DEQ_WORKER_TRACE_WITHOUT_REPORTER";
    if std::env::var_os(CHILD).is_some() {
        let span = Span::root("parent", SpanContext::random());
        let _parent = span.set_local_parent();
        let trace = WorkerTrace::new("inactive");
        assert!(trace.active.is_none());
        trace.property("unused", || panic!("no reporter must mean no formatting"));
        return;
    }
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "decoder::worker_trace::tests::sampled_parent_without_reporter_does_not_start_worker_timing",
        ])
        .env(CHILD, "1")
        .status()
        .unwrap();
    assert!(status.success());
}
