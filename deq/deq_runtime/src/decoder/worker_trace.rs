//! Optional sampled or explicitly enabled lossless diagnostics; inactive traces do no timing work.

pub(crate) fn sampled_pipeline_gid(gid: u64) -> bool {
    gid.wrapping_mul(0xD6E8_FEB8_6659_FD93).rotate_right(16).trailing_zeros() >= 6
}

#[cfg(feature = "cli")]
use crate::misc::fastrace::{Event, Span, SpanContext};

pub(crate) struct WorkerTrace {
    #[cfg(feature = "cli")]
    active: Option<ActiveWorkerTrace>,
}

#[cfg(feature = "cli")]
struct ActiveWorkerTrace {
    span: Span,
    direct: Option<crate::misc::fastrace::LatencySpan>,
    started: std::time::Instant,
    emit_events: bool,
    // Keep properties on the root record: separate fastrace property records can be dropped under load.
    properties: Vec<(String, String)>,
}

#[cfg(feature = "cli")]
impl ActiveWorkerTrace {
    fn diagnostic(name: &'static str) -> Option<Self> {
        if !crate::misc::fastrace::system_diagnostics_enabled() {
            return None;
        }
        let direct = crate::misc::fastrace::LatencySpan::new(name)?;
        Some(Self {
            started: direct.started(),
            span: Span::noop(),
            direct: Some(direct),
            emit_events: false,
            properties: Vec::new(),
        })
    }
}

#[cfg(feature = "cli")]
fn worker_sample_rate() -> u64 {
    static RATE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *RATE.get_or_init(|| {
        let rate = std::env::var_os("DEQ_TRACE_WORKER_SAMPLE_RATE").map_or(1, |value| {
            value
                .to_str()
                .expect("DEQ_TRACE_WORKER_SAMPLE_RATE must be UTF-8")
                .parse::<u64>()
                .expect("DEQ_TRACE_WORKER_SAMPLE_RATE must be a positive integer")
        });
        assert!(rate > 0, "DEQ_TRACE_WORKER_SAMPLE_RATE must be positive");
        rate
    })
}

impl WorkerTrace {
    pub(crate) fn is_active(&self) -> bool {
        #[cfg(feature = "cli")]
        {
            self.active.is_some()
        }
        #[cfg(not(feature = "cli"))]
        {
            false
        }
    }

    pub(crate) fn root(name: &'static str, enabled: bool) -> Self {
        #[cfg(feature = "cli")]
        {
            if enabled && let Some(active) = ActiveWorkerTrace::diagnostic(name) {
                return Self { active: Some(active) };
            }
            let active = (enabled && crate::misc::fastrace::recording_enabled()).then(|| ActiveWorkerTrace {
                span: Span::root(name, SpanContext::random()),
                direct: None,
                started: std::time::Instant::now(),
                emit_events: false,
                properties: Vec::new(),
            });
            Self { active }
        }
        #[cfg(not(feature = "cli"))]
        {
            let _ = (name, enabled);
            Self {}
        }
    }

    pub(super) fn new(name: &'static str) -> Self {
        #[cfg(feature = "cli")]
        {
            if crate::misc::fastrace::system_diagnostics_enabled() {
                static SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let rate = worker_sample_rate();
                let selected = SEQUENCE
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    .is_multiple_of(rate);
                let mut active = selected.then(|| ActiveWorkerTrace::diagnostic(name)).flatten();
                if let Some(trace) = active.as_mut() {
                    trace.properties.push(("worker_sample_rate".into(), rate.to_string()));
                }
                return Self { active };
            }
            let active = crate::misc::fastrace::recording_enabled()
                .then(SpanContext::current_local_parent)
                .flatten()
                .filter(|parent| parent.sampled)
                // A worker can outlive cancellation of its request.
                .map(|parent| ActiveWorkerTrace {
                    span: Span::root(name, parent),
                    direct: None,
                    started: std::time::Instant::now(),
                    emit_events: true,
                    properties: Vec::new(),
                });
            Self { active }
        }
        #[cfg(not(feature = "cli"))]
        {
            let _ = name;
            Self {}
        }
    }

    pub(crate) fn event(&mut self, name: &'static str) {
        if self.is_active() {
            self.event_at(name, std::time::Instant::now());
            self.emit_event(name);
        }
    }

    pub(crate) fn event_at(&mut self, name: &'static str, instant: std::time::Instant) {
        #[cfg(feature = "cli")]
        if let Some(active) = &mut self.active {
            let offset = instant.duration_since(active.started).as_nanos();
            active.properties.push((format!("{name}_offset_ns"), offset.to_string()));
        }
        #[cfg(not(feature = "cli"))]
        let _ = (self, name, instant);
    }

    fn emit_event(&self, name: &'static str) {
        #[cfg(feature = "cli")]
        if let Some(active) = &self.active
            && active.emit_events
        {
            active.span.add_event(Event::new(name));
        }
        #[cfg(not(feature = "cli"))]
        let _ = (self, name);
    }

    pub(crate) fn property(&mut self, name: &'static str, value: impl FnOnce() -> String) {
        #[cfg(feature = "cli")]
        if let Some(active) = &mut self.active {
            active.properties.push((name.to_string(), value()));
        }
        #[cfg(not(feature = "cli"))]
        let _ = (self, name, value);
    }

    pub(super) fn solved(&mut self, success: bool) {
        self.event("solve_finished");
        self.property("solver_success", || success.to_string());
    }

    /// Measure the backend call, excluding validation, reset, and executor queues.
    pub(super) fn measure_solver<R>(&mut self, solve: impl FnOnce() -> R) -> R {
        if !self.is_active() {
            return solve();
        }
        self.resources("solver_start");
        self.emit_event("solver_started");
        let started = std::time::Instant::now();
        let result = solve();
        let returned = std::time::Instant::now();
        self.emit_event("solver_returned");
        self.resources("solver_end");
        self.event_at("solver_started", started);
        self.event_at("solver_returned", returned);
        result
    }

    pub(crate) fn resources(&mut self, stage: &'static str) {
        #[cfg(all(feature = "cli", target_os = "linux"))]
        if let Some(active) = &mut self.active {
            // getrusage writes a plain C record and samples only this worker thread.
            let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
            if unsafe { libc::getrusage(libc::RUSAGE_THREAD, &raw mut usage) } != 0 {
                active
                    .properties
                    .push((format!("{stage}_resource_error"), std::io::Error::last_os_error().to_string()));
                return;
            }
            let user = i128::from(usage.ru_utime.tv_sec) * 1_000_000 + i128::from(usage.ru_utime.tv_usec);
            let system = i128::from(usage.ru_stime.tv_sec) * 1_000_000 + i128::from(usage.ru_stime.tv_usec);
            active.properties.extend([
                (format!("{stage}_thread"), format!("{:?}", std::thread::current().id())),
                (format!("{stage}_user_us"), user.to_string()),
                (format!("{stage}_system_us"), system.to_string()),
                (format!("{stage}_minor_faults"), usage.ru_minflt.to_string()),
                (format!("{stage}_major_faults"), usage.ru_majflt.to_string()),
                (format!("{stage}_voluntary_switches"), usage.ru_nvcsw.to_string()),
                (format!("{stage}_involuntary_switches"), usage.ru_nivcsw.to_string()),
            ]);
        }
        #[cfg(not(all(feature = "cli", target_os = "linux")))]
        let _ = stage;
    }

    pub(super) fn finish(self) {
        drop(self);
    }
}

impl Drop for WorkerTrace {
    fn drop(&mut self) {
        #[cfg(feature = "cli")]
        if let Some(active) = self.active.take() {
            if let Some(mut direct) = active.direct {
                for (key, value) in active.properties {
                    direct.property(key, value);
                }
            } else {
                drop(active.span.with_properties(|| active.properties));
            }
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/worker_trace_test.rs"]
mod tests;
