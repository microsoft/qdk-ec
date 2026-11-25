use super::*;

#[cfg(feature = "cli")]
struct PausingReporter {
    reporter: fastrace::collector::TestReporter,
    paused: Option<std::sync::mpsc::Sender<()>>,
    resume: std::sync::mpsc::Receiver<()>,
}

#[cfg(feature = "cli")]
impl fastrace::collector::Reporter for PausingReporter {
    fn report(&mut self, spans: Vec<fastrace::collector::SpanRecord>) {
        if let Some(paused) = self.paused.take() {
            paused.send(()).unwrap();
            self.resume
                .recv_timeout(std::time::Duration::from_secs(30))
                .expect("test must resume the collector");
        }
        self.reporter.report(spans);
    }
}

#[cfg(feature = "cli")]
#[test]
fn timing_properties_survive_an_overflowing_event_queue() {
    const CHILD: &str = "DEQ_WORKER_TRACE_OVERFLOW";
    if std::env::var_os(CHILD).is_some() {
        let (reporter, records) = fastrace::collector::TestReporter::new();
        let (paused_sender, paused) = std::sync::mpsc::channel();
        let (resume, resume_receiver) = std::sync::mpsc::channel();
        crate::misc::fastrace::set_reporter(
            PausingReporter {
                reporter,
                paused: Some(paused_sender),
                resume: resume_receiver,
            },
            fastrace::collector::Config::default().report_interval(std::time::Duration::from_hours(1)),
        );
        let flushing = std::thread::spawn(fastrace::flush);
        paused
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("collector must pause before filling the event queue");
        let parent = Span::root("parent", SpanContext::random());
        let _parent = parent.set_local_parent();
        let mut trace = WorkerTrace::new("overflow_worker");
        // A paused collector forces overflow; exiting the producer discards its pending records.
        trace = std::thread::spawn(move || {
            for _ in 0..100_000 {
                trace.active.as_ref().unwrap().span.add_event(Event::new("filler"));
            }
            trace.event("worker_started");
            trace.event("initialized");
            trace.solved(true);
            trace.event("worker_finished");
            trace
        })
        .join()
        .unwrap();
        // Publish the root on an uncongested sender: flush cannot drain another sender's backlog.
        trace.finish();
        resume.send(()).unwrap();
        flushing.join().unwrap();
        fastrace::flush();
        let records = records.lock();
        let worker = records
            .iter()
            .find(|record| record.name == "overflow_worker")
            .expect("finished worker root must reach the reporter");
        assert!(worker.events.iter().any(|event| event.name == "filler"));
        assert!(
            !worker.events.iter().any(|event| event.name == "worker_finished"),
            "event queue must overflow before timing is recorded",
        );
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
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "decoder::worker_trace::tests::timing_properties_survive_an_overflowing_event_queue",
        ])
        .env(CHILD, "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "child exited with {}:\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
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
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "decoder::worker_trace::tests::sampled_parent_without_reporter_does_not_start_worker_timing",
        ])
        .env(CHILD, "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "child exited with {}:\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}
