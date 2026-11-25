//! Tracing utilities for performance profiling.
//!
//! The `cli` feature enables fastrace spans and events; `FileReporter` exports them.
//! Register reporters through `set_reporter` to enable optional decoder worker timing.
//! Without `cli`, span and event shims discard tracing calls.

#[cfg(feature = "cli")]
mod inner {
    use fastrace::collector::Reporter;
    use prost::Message;
    use std::fs::File;
    use std::io::{self, BufWriter, Write};
    use std::sync::{Arc, Mutex, Weak, mpsc};
    use std::thread::JoinHandle;

    include!("../proto/deq.misc.fastrace.rs");

    enum WriteCommand {
        Spans(Vec<fastrace::prelude::SpanRecord>),
        Record(SpanRecord),
        Flush(mpsc::SyncSender<()>),
    }

    type FileSender = mpsc::SyncSender<WriteCommand>;
    static FILE_WRITER: Mutex<Weak<FileWriter>> = Mutex::new(Weak::new());

    struct FileWriter {
        sender: Option<FileSender>,
        worker: Option<JoinHandle<io::Result<()>>>,
        failed: Arc<std::sync::atomic::AtomicBool>,
    }

    impl FileWriter {
        fn send(&self, command: WriteCommand) -> Result<(), mpsc::SendError<WriteCommand>> {
            let result = self.sender.as_ref().unwrap().send(command);
            if result.is_err() {
                self.failed.store(true, std::sync::atomic::Ordering::Release);
            }
            result
        }
    }

    impl Drop for FileWriter {
        fn drop(&mut self) {
            drop(self.sender.take());
            if let Some(worker) = self.worker.take()
                && worker.join().is_err()
            {
                eprintln!("timing trace writer panicked");
            }
        }
    }

    /// Required timing records bypass best-effort collector queues and share one monotonic clock.
    pub(crate) struct LatencySpan {
        sender: Arc<FileWriter>,
        started: std::time::Instant,
        record: Option<SpanRecord>,
    }

    impl LatencySpan {
        pub(crate) fn new(name: &str) -> Option<Self> {
            if !super::latency_only() {
                return None;
            }
            let sender = FILE_WRITER.lock().unwrap().upgrade()?;
            Some(Self::with_sender(name, sender))
        }

        fn with_sender(name: &str, sender: Arc<FileWriter>) -> Self {
            static ANCHOR: std::sync::OnceLock<(std::time::Instant, u64)> = std::sync::OnceLock::new();
            assert!(
                !sender.failed.load(std::sync::atomic::Ordering::Acquire),
                "latency trace writer failed"
            );
            let (origin, epoch) = ANCHOR.get_or_init(|| {
                let origin = std::time::Instant::now();
                let epoch = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("system clock precedes the Unix epoch");
                (origin, u64::try_from(epoch.as_nanos()).expect("Unix timestamp fits uint64"))
            });
            let started = std::time::Instant::now();
            let offset = u64::try_from(started.duration_since(*origin).as_nanos()).unwrap();
            Self {
                sender,
                started,
                record: Some(SpanRecord {
                    begin_time_unix_ns: epoch.checked_add(offset).expect("trace timestamp overflow"),
                    name: name.to_owned(),
                    properties: vec![Property {
                        key: "timing_transport".to_owned(),
                        value: "lossless".to_owned(),
                    }],
                    ..Default::default()
                }),
            }
        }

        pub(crate) fn started(&self) -> std::time::Instant {
            self.started
        }

        pub(crate) fn property(&mut self, key: impl Into<String>, value: impl Into<String>) {
            self.record.as_mut().unwrap().properties.push(Property {
                key: key.into(),
                value: value.into(),
            });
        }

        pub(crate) fn event_at(&mut self, name: &str, instant: std::time::Instant) {
            let record = self.record.as_mut().unwrap();
            let offset = u64::try_from(instant.duration_since(self.started).as_nanos()).unwrap();
            let timestamp = record
                .begin_time_unix_ns
                .checked_add(offset)
                .expect("event timestamp overflow");
            let previous = record
                .events
                .last()
                .map_or(record.begin_time_unix_ns, |event| event.timestamp_unix_ns);
            record.events.push(EventRecord {
                name: name.to_owned(),
                timestamp_unix_ns: timestamp,
                delta_ns: timestamp.checked_sub(previous).expect("latency events must be chronological"),
                ..Default::default()
            });
        }

        pub(crate) fn cancel(&mut self) {
            self.record = None;
        }
    }

    impl Drop for LatencySpan {
        fn drop(&mut self) {
            if let Some(mut record) = self.record.take() {
                record.duration_ns = u64::try_from(self.started.elapsed().as_nanos()).unwrap();
                if self.sender.send(WriteCommand::Record(record)).is_err() {
                    eprintln!("latency trace writer stopped before accepting a required record");
                }
            }
        }
    }

    pub struct FileReporter {
        writer: Arc<FileWriter>,
        latency_only: bool,
    }

    impl FileReporter {
        #[must_use]
        pub fn new(file_path: &str) -> Self {
            Self::with_options(file_path, super::latency_only(), super::file_size_limit())
        }

        fn with_options(file_path: &str, latency_only: bool, maximum_bytes: Option<u64>) -> Self {
            let file = File::create(file_path).unwrap();
            // A latency wave can publish thousands of readouts before the writer is scheduled.
            let capacity = if latency_only { 16_384 } else { 64 };
            let (sender, receiver) = mpsc::sync_channel(capacity);
            let failed = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let worker_failed = Arc::clone(&failed);
            let worker = std::thread::Builder::new()
                .name("deq-trace-writer".into())
                .spawn(move || {
                    let result = write_records(file, receiver, latency_only, maximum_bytes);
                    if let Err(error) = &result {
                        worker_failed.store(true, std::sync::atomic::Ordering::Release);
                        eprintln!("timing trace writer failed: {error}");
                    }
                    result
                })
                .unwrap();
            let writer = Arc::new(FileWriter {
                sender: Some(sender),
                worker: Some(worker),
                failed,
            });
            *FILE_WRITER.lock().unwrap() = Arc::downgrade(&writer);
            Self { writer, latency_only }
        }
    }

    impl Reporter for FileReporter {
        fn report(&mut self, mut spans: Vec<fastrace::prelude::SpanRecord>) {
            if self.latency_only {
                spans.retain(|span| match span.name.as_ref() {
                    "static_simulator" | "jit_static_simulator" => true,
                    "decode_window" => !span
                        .properties
                        .iter()
                        .any(|(key, value)| key == "readout_count" && value == "0"),
                    _ => false,
                });
            }
            self.writer
                .send(WriteCommand::Spans(spans))
                .expect("timing trace writer stopped");
        }
    }

    fn write_records(
        file: File,
        receiver: mpsc::Receiver<WriteCommand>,
        latency_only: bool,
        maximum_bytes: Option<u64>,
    ) -> io::Result<()> {
        let mut writer = super::LimitedWriter::new(BufWriter::new(file), maximum_bytes);
        let mut buffer = Vec::new();
        for command in receiver {
            match command {
                WriteCommand::Spans(spans) => {
                    for span in spans {
                        let mut record = convert(span);
                        if latency_only {
                            record.properties.retain(|property| {
                                matches!(
                                    property.key.as_str(),
                                    "window" | "commit_region" | "readout_count" | "shot" | "error"
                                ) || property.key.ends_with("_offset_ns")
                            });
                            record
                                .events
                                .retain(|event| matches!(event.name.as_str(), "start_decoding" | "process_result"));
                        }
                        let content = TraceFile { spans: vec![record] };
                        buffer.clear();
                        content.encode(&mut buffer).unwrap();
                        writer.write_all(&buffer)?;
                    }
                }
                WriteCommand::Record(record) => {
                    buffer.clear();
                    TraceFile { spans: vec![record] }.encode(&mut buffer).unwrap();
                    writer.write_all(&buffer)?;
                }
                WriteCommand::Flush(completed) => {
                    writer.flush()?;
                    let _ = completed.send(());
                }
            }
        }
        writer.flush()
    }

    pub(super) fn flush_file_writer() {
        let writer = FILE_WRITER.lock().unwrap().upgrade();
        if let Some(sender) = writer {
            flush_sender(&sender);
        }
    }

    fn flush_sender(sender: &FileWriter) {
        let (completed, flushed) = mpsc::sync_channel(0);
        sender
            .send(WriteCommand::Flush(completed))
            .expect("timing trace writer stopped before flush");
        flushed.recv().expect("timing trace writer failed to flush");
    }

    fn convert(span: fastrace::prelude::SpanRecord) -> SpanRecord {
        let mut last_time = span.begin_time_unix_ns;
        SpanRecord {
            begin_time_unix_ns: span.begin_time_unix_ns,
            duration_ns: span.duration_ns,
            name: span.name.into_owned(),
            properties: std::iter::once(Property {
                key: "trace_id".to_string(),
                value: span.trace_id.to_string(),
            })
            .chain(span.properties.into_iter().map(|(key, value)| Property {
                key: key.into_owned(),
                value: value.into_owned(),
            }))
            .collect(),
            events: span
                .events
                .into_iter()
                .map(|e| {
                    let delta_ns = e.timestamp_unix_ns.saturating_sub(last_time);
                    last_time = e.timestamp_unix_ns;
                    EventRecord {
                        name: e.name.into_owned(),
                        timestamp_unix_ns: e.timestamp_unix_ns,
                        properties: e
                            .properties
                            .into_iter()
                            .map(|(key, value)| Property {
                                key: key.into_owned(),
                                value: value.into_owned(),
                            })
                            .collect(),
                        delta_ns,
                    }
                })
                .collect(),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn system_diagnostic_roots_use_lossless_transport_without_a_parent() {
            const CHILD: &str = "DEQ_TEST_LOSSLESS_SYSTEM_TRACE";
            if std::env::var_os(CHILD).is_some() {
                let output = tempfile::NamedTempFile::new().unwrap();
                let reporter = FileReporter::new(output.path().to_str().unwrap());
                crate::misc::fastrace::set_reporter(
                    reporter,
                    fastrace::collector::Config::default().report_interval(std::time::Duration::from_hours(1)),
                );
                let mut disabled = crate::decoder::worker_trace::WorkerTrace::root("disabled", false);
                disabled.property("unused", || panic!("disabled diagnostics must not format properties"));
                let mut trace = crate::decoder::worker_trace::WorkerTrace::root("diagnostic_worker", true);
                assert!(trace.is_active());
                trace.event("admitted");
                std::thread::spawn(move || {
                    trace.event("solver_started");
                    trace.event("solver_returned");
                    trace.property("solver_success", || "true".into());
                })
                .join()
                .unwrap();
                crate::misc::fastrace::flush();
                let records = TraceFile::decode(std::fs::read(output.path()).unwrap().as_slice()).unwrap();
                let [record] = records.spans.as_slice() else {
                    panic!("expected exactly one enabled diagnostic record");
                };
                assert_eq!(record.name, "diagnostic_worker");
                assert!(
                    record
                        .properties
                        .iter()
                        .any(|property| { property.key == "timing_transport" && property.value == "lossless" })
                );
                let offsets: Vec<u64> = ["admitted", "solver_started", "solver_returned"]
                    .into_iter()
                    .map(|name| {
                        record
                            .properties
                            .iter()
                            .find(|property| property.key == format!("{name}_offset_ns"))
                            .unwrap()
                            .value
                            .parse()
                            .unwrap()
                    })
                    .collect();
                assert!(offsets.windows(2).all(|pair| pair[0] <= pair[1]));
                assert!(offsets[2] <= record.duration_ns);
                return;
            }
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "misc::fastrace::inner::tests::system_diagnostic_roots_use_lossless_transport_without_a_parent",
                ])
                .env(CHILD, "1")
                .env("DEQ_TRACE_MODE", "latency")
                .env("DEQ_TRACE_DIAGNOSTICS", "1")
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
        fn outstanding_latency_spans_keep_the_writer_alive_without_blocking_reporter_drop() {
            let output = tempfile::NamedTempFile::new().unwrap();
            let path = output.path().to_owned();
            let (spans, received) = mpsc::sync_channel(0);
            let (finished, dropped) = mpsc::sync_channel(1);
            let owner = std::thread::spawn(move || {
                let reporter = FileReporter::with_options(path.to_str().unwrap(), true, None);
                let mut span = LatencySpan::with_sender("static_simulator", Arc::clone(&reporter.writer));
                span.property("shot", "0");
                assert!(spans.send(span).is_ok());
                drop(reporter);
                finished.send(()).unwrap();
            });
            let span = received.recv().unwrap();
            let released = dropped.recv_timeout(std::time::Duration::from_secs(1)).is_ok();
            drop(span);
            owner.join().unwrap();
            assert!(released, "reporter replacement must not wait for a still-live span");
            let trace = TraceFile::decode(std::fs::read(output.path()).unwrap().as_slice()).unwrap();
            assert_eq!(trace.spans.len(), 1, "the final producer must flush its record");
        }

        #[test]
        fn writer_failure_is_visible_without_panicking_from_span_drop() {
            let output = tempfile::NamedTempFile::new().unwrap();
            let reporter = FileReporter::with_options(output.path().to_str().unwrap(), true, Some(0));
            let first = LatencySpan::with_sender("static_simulator", Arc::clone(&reporter.writer));
            let outstanding = LatencySpan::with_sender("static_simulator", Arc::clone(&reporter.writer));
            drop(first);
            let started = std::time::Instant::now();
            while !reporter.writer.failed.load(std::sync::atomic::Ordering::Acquire) {
                assert!(started.elapsed() < std::time::Duration::from_secs(2));
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| drop(outstanding))).is_ok());
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    flush_sender(&reporter.writer);
                }))
                .is_err()
            );
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    LatencySpan::with_sender("static_simulator", Arc::clone(&reporter.writer))
                }))
                .is_err()
            );
        }

        #[test]
        fn required_latency_records_survive_concurrent_bursts() {
            let output = tempfile::NamedTempFile::new().unwrap();
            let reporter = FileReporter::with_options(output.path().to_str().unwrap(), true, Some(10_000_000));
            let handles: Vec<_> = (0..4)
                .map(|worker| {
                    let sender = Arc::clone(&reporter.writer);
                    std::thread::spawn(move || {
                        for index in 0..4000 {
                            let mut span = LatencySpan::with_sender("static_simulator", Arc::clone(&sender));
                            span.property("shot", (worker * 4000 + index).to_string());
                            span.event_at("start_decoding", std::time::Instant::now());
                            span.event_at("process_result", std::time::Instant::now());
                        }
                    })
                })
                .collect();
            for handle in handles {
                handle.join().unwrap();
            }
            flush_sender(&reporter.writer);
            let trace = TraceFile::decode(std::fs::read(output.path()).unwrap().as_slice()).unwrap();
            assert_eq!(trace.spans.len(), 16_000);
            let shots: std::collections::HashSet<_> = trace
                .spans
                .iter()
                .map(|span| {
                    span.properties
                        .iter()
                        .find(|property| property.key == "shot")
                        .unwrap()
                        .value
                        .clone()
                })
                .collect();
            assert_eq!(shots.len(), 16_000);
            for span in trace.spans {
                assert_eq!(span.events.len(), 2);
                assert!(span.begin_time_unix_ns <= span.events[0].timestamp_unix_ns);
                assert!(span.events[0].timestamp_unix_ns <= span.events[1].timestamp_unix_ns);
                assert!(span.events[1].timestamp_unix_ns <= span.begin_time_unix_ns + span.duration_ns);
            }
        }

        #[test]
        fn latency_reporter_retains_every_readout_window_but_not_payloads() {
            let output = tempfile::NamedTempFile::new().unwrap();
            let mut reporter = FileReporter::with_options(output.path().to_str().unwrap(), true, Some(10_000));
            let window = |readouts: &str| fastrace::collector::SpanRecord {
                name: "decode_window".into(),
                properties: vec![
                    ("readout_count".into(), readouts.to_owned().into()),
                    ("window".into(), "{1, 2}".into()),
                    ("commit_region".into(), "{2}".into()),
                    ("pauli_frame_updated_offset_ns".into(), "7".into()),
                    ("parity_factor".into(), "large payload".into()),
                ],
                ..Default::default()
            };
            reporter.report(vec![
                window("0"),
                window("2"),
                fastrace::collector::SpanRecord {
                    name: "decoder_job".into(),
                    ..Default::default()
                },
                fastrace::collector::SpanRecord {
                    name: "static_simulator".into(),
                    properties: vec![("shot".into(), "0".into()), ("sample".into(), "payload".into())],
                    events: vec![fastrace::collector::EventRecord {
                        name: "start_decoding".into(),
                        timestamp_unix_ns: 100,
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            ]);
            flush_sender(&reporter.writer);
            let trace = TraceFile::decode(std::fs::read(output.path()).unwrap().as_slice()).unwrap();
            assert_eq!(trace.spans.len(), 2);
            assert_eq!(trace.spans[0].properties.len(), 4);
            assert_eq!(trace.spans[1].properties.len(), 1);
            assert_eq!(trace.spans[1].events[0].timestamp_unix_ns, 100);
        }

        #[test]
        fn file_limit_rejects_a_record_before_exceeding_the_limit() {
            let mut bytes = vec![];
            {
                let mut writer = super::super::LimitedWriter::new(&mut bytes, Some(4));
                writer.write_all(&[1, 2, 3]).unwrap();
                assert!(writer.write_all(&[4, 5]).is_err());
                writer.flush().unwrap();
            }
            assert_eq!(bytes, [1, 2, 3]);
        }

        #[test]
        fn buffered_reporter_flushes_all_batches_and_owned_metadata() {
            let output = tempfile::NamedTempFile::new().unwrap();
            let mut reporter = FileReporter::new(output.path().to_str().unwrap());
            for batch in 0..128 {
                reporter.report(vec![
                    fastrace::collector::SpanRecord {
                        begin_time_unix_ns: 100,
                        duration_ns: 20,
                        name: format!("batch-{batch}").into(),
                        properties: vec![("owned-key".to_string().into(), batch.to_string().into())],
                        events: vec![fastrace::collector::EventRecord {
                            name: "ready".to_string().into(),
                            timestamp_unix_ns: 105,
                            properties: vec![("stage".to_string().into(), "committed".to_string().into())],
                        }],
                        ..Default::default()
                    };
                    128
                ]);
            }
            flush_sender(&reporter.writer);
            let contents = std::fs::read(output.path()).unwrap();
            let trace = TraceFile::decode(contents.as_slice()).unwrap();
            assert_eq!(trace.spans.len(), 128 * 128);
            for (batch, spans) in trace.spans.chunks(128).enumerate() {
                for span in spans {
                    assert_eq!(span.name, format!("batch-{batch}"));
                    assert_eq!(span.properties[1].value, batch.to_string());
                    assert_eq!(span.events[0].name, "ready");
                    assert_eq!(span.events[0].delta_ns, 5);
                    assert_eq!(span.events[0].properties[0].value, "committed");
                }
            }
        }
    }
}

#[cfg(feature = "cli")]
pub use inner::FileReporter;
#[cfg(feature = "cli")]
pub(crate) use inner::LatencySpan;

pub(crate) fn file_size_limit() -> Option<u64> {
    std::env::var_os("DEQ_TRACE_MAX_BYTES").map(|value| {
        value
            .to_str()
            .expect("DEQ_TRACE_MAX_BYTES must be UTF-8")
            .parse::<u64>()
            .expect("DEQ_TRACE_MAX_BYTES must be an unsigned integer")
    })
}

pub(crate) struct LimitedWriter<W> {
    writer: W,
    written: u64,
    maximum_bytes: Option<u64>,
}

impl<W> LimitedWriter<W> {
    pub(crate) fn new(writer: W, maximum_bytes: Option<u64>) -> Self {
        Self {
            writer,
            written: 0,
            maximum_bytes,
        }
    }
}

impl<W: std::io::Write> std::io::Write for LimitedWriter<W> {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        let total = self
            .written
            .checked_add(u64::try_from(data.len()).unwrap())
            .ok_or_else(|| std::io::Error::other("trace length overflow"))?;
        if self.maximum_bytes.is_some_and(|limit| total > limit) {
            return Err(std::io::Error::other("trace exceeded DEQ_TRACE_MAX_BYTES"));
        }
        let written = self.writer.write(data)?;
        self.written += u64::try_from(written).unwrap();
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

pub(crate) fn latency_only() -> bool {
    static LATENCY_ONLY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *LATENCY_ONLY.get_or_init(|| match std::env::var_os("DEQ_TRACE_MODE") {
        None => false,
        Some(value) if value == "full" => false,
        Some(value) if value == "latency" => true,
        Some(_) => panic!("DEQ_TRACE_MODE must be full or latency"),
    })
}

pub(crate) fn system_diagnostics_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    latency_only()
        && *ENABLED.get_or_init(|| {
            std::env::var_os("DEQ_TRACE_DIAGNOSTICS").is_some_and(|value| {
                assert!(value == "0" || value == "1", "DEQ_TRACE_DIAGNOSTICS must be 0 or 1");
                value == "1"
            })
        })
}

#[cfg(feature = "cli")]
static RECORDING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Install the reporter and enable optional worker timing. Use this instead of fastrace's setter.
#[cfg(feature = "cli")]
pub fn set_reporter(reporter: impl fastrace::collector::Reporter, config: fastrace::collector::Config) {
    fastrace::set_reporter(reporter, config);
    RECORDING.store(true, std::sync::atomic::Ordering::Release);
}

/// Flush collected spans and wait for queued file writes.
///
/// # Panics
/// Panics if the file writer stopped or could not flush its output.
#[cfg(feature = "cli")]
pub fn flush() {
    fastrace::flush();
    inner::flush_file_writer();
}

#[cfg(feature = "cli")]
pub(crate) fn recording_enabled() -> bool {
    RECORDING.load(std::sync::atomic::Ordering::Acquire) && !latency_only()
}

// Re-export or provide no-op shims so call sites compile without cfg blocks.
#[cfg(feature = "cli")]
pub use fastrace::prelude::{Event, Span, SpanContext};

/// No-op span that silently discards all property and event calls.
#[cfg(not(feature = "cli"))]
pub struct Span;

#[cfg(not(feature = "cli"))]
impl Span {
    #[inline]
    pub fn noop() -> Self {
        Self
    }
    #[inline]
    pub fn root(_name: &'static str, _context: SpanContext) -> Self {
        Self
    }
    #[inline]
    pub fn add_property<K: Into<String>, V: Into<String>, F: FnOnce() -> (K, V)>(&self, _f: F) {}
    #[inline]
    pub fn add_event(&self, _event: Event) {}
}

/// No-op event.
#[cfg(not(feature = "cli"))]
pub struct Event;

#[cfg(not(feature = "cli"))]
impl Event {
    #[inline]
    pub fn new(_name: &'static str) -> Self {
        Self
    }
    #[inline]
    pub fn with_property<K: Into<String>, V: Into<String>, F: FnOnce() -> (K, V)>(self, _f: F) -> Self {
        self
    }
}

/// No-op span context.
#[cfg(not(feature = "cli"))]
pub struct SpanContext;

#[cfg(not(feature = "cli"))]
impl SpanContext {
    #[inline]
    pub fn random() -> Self {
        Self
    }
}
