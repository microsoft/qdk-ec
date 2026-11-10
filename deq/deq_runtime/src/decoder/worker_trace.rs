//! Optional worker telemetry; no clocks or property formatting without a sampled trace and reporter.

#[cfg(feature = "cli")]
use crate::misc::fastrace::{Event, Span, SpanContext};

pub(super) struct WorkerTrace {
    #[cfg(feature = "cli")]
    active: Option<(Span, std::time::Instant)>,
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
                .map(|parent| (Span::root(name, parent), std::time::Instant::now()));
            Self { active }
        }
        #[cfg(not(feature = "cli"))]
        {
            let _ = name;
            Self {}
        }
    }

    pub(super) fn event(&self, name: &'static str) {
        #[cfg(feature = "cli")]
        if let Some((span, started)) = &self.active {
            let offset = started.elapsed().as_nanos();
            span.add_property(|| (format!("{name}_offset_ns"), offset.to_string()));
            span.add_event(Event::new(name));
        }
        #[cfg(not(feature = "cli"))]
        let _ = (self, name);
    }

    pub(super) fn property(&self, name: &'static str, value: impl FnOnce() -> String) {
        #[cfg(feature = "cli")]
        if let Some((span, _)) = &self.active {
            span.add_property(|| (name, value()));
        }
        #[cfg(not(feature = "cli"))]
        let _ = (self, name, value);
    }

    pub(super) fn solved(&self, success: bool) {
        self.event("solve_finished");
        self.property("solver_success", || success.to_string());
    }

    pub(super) fn finish(self) {
        #[cfg(feature = "cli")]
        drop(self.active);
        #[cfg(not(feature = "cli"))]
        let _ = self;
    }
}

#[cfg(test)]
#[path = "../../tests/unit/worker_trace_test.rs"]
mod tests;
