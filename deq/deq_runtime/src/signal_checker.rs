//! Signal checker for graceful Ctrl+C handling.
//!
//! When the `cli` feature is enabled, uses the `ctrlc` crate to install a
//! native signal handler. When disabled, `check()` always returns `Ok(())`.

use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, Ordering};

pub static INTERRUPTED: AtomicBool = AtomicBool::new(false);

pub static SIGNAL_CHECKER: LazyLock<SignalChecker> = LazyLock::new(SignalChecker::new);

pub struct SignalChecker {
    _private: (),
}

impl SignalChecker {
    pub fn new() -> Self {
        #[cfg(feature = "cli")]
        ctrlc::set_handler(|| {
            INTERRUPTED.store(true, Ordering::SeqCst);
        })
        .expect("Failed to set Ctrl+C handler");
        Self { _private: () }
    }

    #[allow(clippy::result_unit_err)]
    #[inline]
    pub fn check(&self) -> Result<(), ()> {
        if INTERRUPTED.load(Ordering::SeqCst) { Err(()) } else { Ok(()) }
    }

    #[inline]
    pub fn reset(&self) {
        INTERRUPTED.store(false, Ordering::SeqCst);
    }
}

impl Default for SignalChecker {
    fn default() -> Self {
        Self::new()
    }
}
