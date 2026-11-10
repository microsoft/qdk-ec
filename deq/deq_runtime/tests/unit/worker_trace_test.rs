use super::*;

#[cfg(feature = "cli")]
#[test]
fn timing_properties_survive_an_overflowing_event_queue() {
    const CHILD: &str = "DEQ_WORKER_TRACE_OVERFLOW";
    if std::env::var_os(CHILD).is_some() {
        let (reporter, records) = fastrace::collector::TestReporter::new();
        crate::misc::fastrace::set_reporter(
            reporter,
            fastrace::collector::Config::default().report_interval(std::time::Duration::from_hours(1)),
        );
        let parent = Span::root("parent", SpanContext::random());
        let _parent = parent.set_local_parent();
        let mut trace = WorkerTrace::new("overflow_worker");
        for _ in 0..100_000 {
            trace.active.as_ref().unwrap().span.add_event(Event::new("filler"));
        }
        trace.event("worker_started");
        trace.event("initialized");
        trace.solved(true);
        trace.event("worker_finished");
        fastrace::flush();
        trace.finish();
        fastrace::flush();
        let records = records.lock();
        let worker = records.iter().find(|record| record.name == "overflow_worker").unwrap();
        for key in [
            "worker_started_offset_ns",
            "initialized_offset_ns",
            "solve_finished_offset_ns",
            "worker_finished_offset_ns",
            "solver_success",
        ] {
            assert!(worker.properties.iter().any(|(name, _)| name == key), "{key}");
        }
        return;
    }
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "decoder::worker_trace::tests::timing_properties_survive_an_overflowing_event_queue",
        ])
        .env(CHILD, "1")
        .status()
        .unwrap();
    assert!(status.success());
}

#[test]
fn inactive_worker_trace_does_not_evaluate_properties() {
    let mut trace = WorkerTrace::new("inactive");
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
    let mut trace = WorkerTrace::new("inactive");
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
        let mut trace = WorkerTrace::new("inactive");
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
