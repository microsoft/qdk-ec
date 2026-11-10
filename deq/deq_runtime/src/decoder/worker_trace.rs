//! Optional worker telemetry; no clocks or property formatting without a sampled trace and reporter.

#[cfg(feature = "cli")]
use crate::misc::fastrace::{Event, Span, SpanContext};

pub(super) struct WorkerTrace {
    #[cfg(feature = "cli")]
    active: Option<ActiveWorkerTrace>,
}

#[cfg(feature = "cli")]
struct ActiveWorkerTrace {
    span: Span,
    started: std::time::Instant,
    // Keep properties on the root record: separate fastrace property records can be dropped under load.
    properties: Vec<(String, String)>,
}

impl WorkerTrace {
    pub(super) fn new(name: &'static str) -> Self {
        #[cfg(feature = "cli")]
        {
            let active = crate::misc::fastrace::recording_enabled()
                .then(SpanContext::current_local_parent)
                .flatten()
                .filter(|parent| parent.sampled)
                // A worker can outlive cancellation of its request.
                .map(|parent| ActiveWorkerTrace {
                    span: Span::root(name, parent),
                    started: std::time::Instant::now(),
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

    pub(super) fn event(&mut self, name: &'static str) {
        #[cfg(feature = "cli")]
        if let Some(active) = &mut self.active {
            let offset = active.started.elapsed().as_nanos();
            active.properties.push((format!("{name}_offset_ns"), offset.to_string()));
            active.span.add_event(Event::new(name));
        }
        #[cfg(not(feature = "cli"))]
        let _ = (self, name);
    }

    pub(super) fn property(&mut self, name: &'static str, value: impl FnOnce() -> String) {
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

    pub(super) fn finish(self) {
        #[cfg(feature = "cli")]
        if let Some(active) = self.active {
            drop(active.span.with_properties(|| active.properties));
        }
        #[cfg(not(feature = "cli"))]
        let _ = self;
    }
}

#[cfg(test)]
#[path = "../../tests/unit/worker_trace_test.rs"]
mod tests;
