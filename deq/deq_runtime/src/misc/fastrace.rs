//! Tracing utilities for performance profiling.
//!
//! When the `fastrace` feature is enabled, spans and events are recorded using
//! the fastrace crate and can be exported to a file via `FileReporter`.
//! When disabled, all tracing calls are no-ops with zero runtime cost.

#[cfg(feature = "cli")]
mod inner {
    use fastrace::collector::Reporter;
    use prost::Message;
    use std::fs::File;
    use std::io::{BufWriter, Write};

    include!("../proto/deq.misc.fastrace.rs");

    pub struct FileReporter {
        buf_writer: BufWriter<std::fs::File>,
        buffer: Vec<u8>,
    }

    impl FileReporter {
        pub fn new(file_path: &str) -> Self {
            let file = File::create(file_path).unwrap();
            let buf_writer = BufWriter::new(file);
            Self {
                buf_writer,
                buffer: vec![],
            }
        }
    }

    impl Reporter for FileReporter {
        fn report(&mut self, spans: Vec<fastrace::prelude::SpanRecord>) {
            for span in spans.iter() {
                let record = convert(span);
                let content = TraceFile { spans: vec![record] };
                self.buffer.clear();
                content.encode(&mut self.buffer).unwrap();
                self.buf_writer.write_all(&self.buffer).unwrap();
            }
            self.buf_writer.flush().unwrap();
        }
    }

    pub fn convert(span: &fastrace::prelude::SpanRecord) -> SpanRecord {
        let mut last_time = span.begin_time_unix_ns;
        SpanRecord {
            begin_time_unix_ns: span.begin_time_unix_ns,
            duration_ns: span.duration_ns,
            name: span.name.to_string(),
            properties: span
                .properties
                .iter()
                .map(|(k, v)| Property {
                    key: k.to_string(),
                    value: v.to_string(),
                })
                .collect(),
            events: span
                .events
                .iter()
                .map(|e| {
                    let delta_ns = e.timestamp_unix_ns.saturating_sub(last_time);
                    last_time = e.timestamp_unix_ns;
                    EventRecord {
                        name: e.name.to_string(),
                        timestamp_unix_ns: e.timestamp_unix_ns,
                        properties: e
                            .properties
                            .iter()
                            .map(|(k, v)| Property {
                                key: k.to_string(),
                                value: v.to_string(),
                            })
                            .collect(),
                        delta_ns,
                    }
                })
                .collect(),
        }
    }
}

#[cfg(feature = "cli")]
pub use inner::FileReporter;

// Re-export or provide no-op shims so call sites compile without cfg blocks.
#[cfg(feature = "cli")]
pub use fastrace::prelude::{Event, Span, SpanContext};

/// No-op span that silently discards all property and event calls.
#[cfg(not(feature = "cli"))]
pub struct Span;

#[cfg(not(feature = "cli"))]
impl Span {
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
