#[cfg(feature = "cli")]
use structdoc::StructDoc;

#[cfg(feature = "cli")]
const STRUCT_DOC_PREFIX: &str = "<root> (Struct)";

#[cfg(feature = "cli")]
pub fn help_message<T: StructDoc>(struct_name: &str) -> String {
    let doc_string = T::document().to_string();
    debug_assert!(
        doc_string.starts_with(STRUCT_DOC_PREFIX),
        "the help message is expected to start with `{}`",
        STRUCT_DOC_PREFIX
    );
    doc_string.replacen(STRUCT_DOC_PREFIX, struct_name, 1)
}

/// Given the probabilities of two independent events A and B, returns the
/// probability that A occurs or B occurs, but not both.
pub fn exclusive_probability_of(probability_a: f64, probability_b: f64) -> f64 {
    probability_a + probability_b - 2.0 * probability_a * probability_b
}

pub fn weight_of(probability: f64) -> f64 {
    debug_assert!(probability > 0.0 && probability < 1.0);
    -(probability / (1.0 - probability)).ln()
}

/// Returns nanoseconds elapsed since the first call to this function in the
/// current process, using a monotonic clock.
///
/// Trace consumers only ever use timestamps relatively (deltas between events
/// on the same shot), so a process-local monotonic origin is sufficient and
/// avoids the cross-core wall-clock skew that can make `SystemTime::now()`
/// non-monotonic on some platforms (notably aarch64 / virtualised CI).
pub fn timestamp_ns() -> u64 {
    static ORIGIN: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let origin = ORIGIN.get_or_init(std::time::Instant::now);
    origin.elapsed().as_nanos() as u64
}

#[cfg(test)]
mod tests {
    #[test]
    #[cfg(all(feature = "cli", feature = "simulator"))]
    fn help_message() {
        use structdoc::StructDoc;
        // cargo test help_message -- --nocapture
        // we assumed that the help message starts with a line of `<root> (Struct):`
        assert!(
            crate::simulator::static_simulator::StaticSimulatorConfig::document()
                .to_string()
                .starts_with("<root> (Struct):")
        );
    }
}
