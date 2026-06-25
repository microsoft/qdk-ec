//! Cache key for the `loaded_decoders` caches in [`WindowCoordinator`] and
//! [`MonolithicCoordinator`].
//!
//! Both coordinators cache `LoadedDecoder` instances keyed by what was used to
//! build the merged decoding hypergraph for a window.  This module owns the
//! key type and the helpers that build it; both coordinators share the same
//! key shape so that a fix to the key (e.g. adding a newly relevant input)
//! lands in one place.
//!
//! The key intentionally lives outside the two coordinator modules — neither
//! coordinator "owns" it conceptually, so placing it here mirrors the
//! placement of other cross-coordinator shared types in
//! [`crate::coordinator`].
//!
//! [`WindowCoordinator`]: crate::coordinator::WindowCoordinator
//! [`MonolithicCoordinator`]: crate::coordinator::MonolithicCoordinator

use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use hashbrown::HashMap;

use crate::bin::error_model_type::RemoteCheckModel;
use crate::bin::{ErrorModel, ErrorModelType, ProbabilityModifier};
use crate::controller::jit_controller::hash_error_model_type_structural;
use crate::misc::relative_program::{RelativeMapping, RelativeProgram};

/// Bit-packed view of [`ProbabilityModifier`] whose `f64` fields are
/// stored as their raw `u64` bit-patterns so the value implements
/// `Hash + Eq`.
///
/// Matches the established pattern in
/// [`crate::controller::jit_controller::hash_error`], where
/// `error.probability.to_bits()` is used for the same reason.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProbabilityModifierBits {
    pub probabilities: Vec<u64>,
    pub sparse_indices: Vec<u64>,
    pub sparse_probabilities: Vec<u64>,
}

impl From<&ProbabilityModifier> for ProbabilityModifierBits {
    fn from(pm: &ProbabilityModifier) -> Self {
        Self {
            probabilities: pm.probabilities.iter().map(|p| p.to_bits()).collect(),
            sparse_indices: pm.sparse_indices.clone(),
            sparse_probabilities: pm.sparse_probabilities.iter().map(|p| p.to_bits()).collect(),
        }
    }
}

/// Fingerprint of all per-`ErrorModel` state read live during
/// `decoding_hypergraph()`.  Captures every field whose value can affect the
/// merged hyperedges of a window:
///
/// - `etype_digest` — a structural hash of the resolved
///   [`ErrorModelType`] (computed via
///   [`hash_error_model_type_structural`]) so the raw `errors` list,
///   `ctype`, and so on contribute to the key.  Replaces the previous
///   raw `etype: u64` field, which was only an ID and did not invalidate
///   the cache when the resolved type changed under the same `etype`.
/// - `probability_modifier` (`pm`) — overrides per-error probabilities.
/// - `remote_check_models` — provides the `check_bias` added to vertex
///   indices for remote checks.  Stored as `Arc<Vec<...>>` (shared with
///   the originating `ErrorModel`) so building a fingerprint is a refcount
///   bump rather than a full vector clone; equality and hashing transparently
///   delegate to the inner `Vec`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ErrorModelFingerprint {
    pub etype_digest: u64,
    pub pm: Option<ProbabilityModifierBits>,
    pub remote_check_models: Arc<Vec<Option<RemoteCheckModel>>>,
}

impl ErrorModelFingerprint {
    /// Build a fingerprint with a precomputed `etype_digest` and an already-`Arc`'d
    /// remote-check-models vector.  The `Arc` is the same one stored on the
    /// coordinator's `ErrorModel`, so this constructor performs at most one
    /// `ProbabilityModifierBits` allocation per call.
    pub fn new(
        instance: &ErrorModel,
        modified_remote_check_models: Arc<Vec<Option<RemoteCheckModel>>>,
        etype_digest: u64,
    ) -> Self {
        let pm = instance
            .modifier
            .as_ref()
            .and_then(|m| m.probability_modifier.as_ref())
            .map(ProbabilityModifierBits::from);
        Self {
            etype_digest,
            pm,
            remote_check_models: modified_remote_check_models,
        }
    }
}

/// Compute the structural digest of a resolved [`ErrorModelType`] for
/// use as the `etype_digest` field of [`ErrorModelFingerprint`].
///
/// Delegates to [`hash_error_model_type_structural`] so the digest matches
/// what [`crate::controller::jit_controller::ErrorModelTypeKey`]'s `Hash`
/// impl produces — i.e. the same structural identity already used by
/// `TypeCache` to deduplicate error-model types across JIT compilations.
pub fn etype_digest(emt: &ErrorModelType) -> u64 {
    let mut hasher = DefaultHasher::new();
    hash_error_model_type_structural(emt, &mut hasher);
    hasher.finish()
}

/// Cache key for `loaded_decoders` in both
/// [`crate::coordinator::WindowCoordinator`] and
/// [`crate::coordinator::MonolithicCoordinator`].
///
/// The merged decoding hypergraph built by `decoding_hypergraph()` depends
/// on three pieces of per-call state, all of which must be in the key:
///
/// 1. `relative_program` — the window structure (gadgets, error-model slots,
///    remote-check projections).
/// 2. `error_model_fingerprints` — the live per-`ErrorModel` modifier state
///    that overrides probabilities and remote-check biases, plus the
///    structural digest of the resolved error-model type.  Required because
///    `error_models` is cleared on every shot and `next_eid` resets to 1,
///    so the same `eid` is rebound across shots — typically to a different
///    modifier — and because the simulator can in principle reload the
///    library between shots and change the type backing a given `etype`.
/// 3. `committing_local_cids` — the per-window commit region (in local-cid
///    indexing).  Required because the same `RelativeProgram` can be decoded
///    with different commit regions in different windows, and the
///    `is_in_commit_region` check at edge-construction time drops hyperedges
///    whose checks straddle the window boundary.  Empty for the
///    `MonolithicCoordinator` (which decodes the entire connected subgraph
///    and has no commit-region concept).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DecoderCacheKey {
    pub relative_program: RelativeProgram,
    pub error_model_fingerprints: Vec<ErrorModelFingerprint>,
    pub committing_local_cids: Vec<u32>,
}

/// Abstraction over the per-coordinator `ErrorModel` wrapper struct so that
/// [`build_modifier_fingerprints`] can run unchanged against either
/// `coordinator::monolithic_coordinator::ErrorModel` or
/// `coordinator::window_coordinator::ErrorModel`.
///
/// Each coordinator stores the resolved error-model instance plus its
/// modified remote-check-models list — the two pieces
/// [`ErrorModelFingerprint::new`] consumes — but the two wrappers carry
/// additional, coordinator-specific fields (the monolithic one holds an
/// extra `watch::Sender` for `expanded_remote_check_models`).  This trait
/// is the seam.
///
/// `modified_remote_check_models` returns the borrowed `Arc` so that the
/// fingerprint constructor can share the underlying `Vec` (refcount bump)
/// instead of cloning it.
pub trait FingerprintSource {
    fn instance(&self) -> &ErrorModel;
    fn modified_remote_check_models(&self) -> &Arc<Vec<Option<RemoteCheckModel>>>;
}

/// Build the per-window modifier fingerprint vector indexed by `local_eid`
/// (i.e. `mapping.global_eid_of[local_eid]` is the global `eid` whose
/// modifier state we fingerprint).  `error_model_types` resolves each
/// `instance.etype` to the [`ErrorModelType`] whose structural digest
/// contributes to the fingerprint.
///
/// Used by both [`crate::coordinator::WindowCoordinator`] and
/// [`crate::coordinator::MonolithicCoordinator`] — the only thing that
/// differs between them is the concrete `E: FingerprintSource` type.
///
/// `etype_digest` is computed at most once per *distinct* `etype` per call
/// (typically only a handful even when `mapping.global_eid_of` is thousands
/// of entries long), and `remote_check_models` is `Arc::clone`'d rather than
/// copied, so the hot-path cost is dominated by the `ProbabilityModifierBits`
/// allocation per eid (if a modifier is present).
pub fn build_modifier_fingerprints<E: FingerprintSource>(
    mapping: &RelativeMapping,
    error_models: &HashMap<u64, E>,
    error_model_types: &HashMap<u64, Arc<ErrorModelType>>,
) -> Vec<ErrorModelFingerprint> {
    let mut digest_cache: HashMap<u64, u64> = HashMap::with_capacity(error_model_types.len());
    mapping
        .global_eid_of
        .iter()
        .map(|eid| {
            let em = error_models.get(eid).expect("error_model present for eid in window");
            let etype = em.instance().etype;
            let digest = *digest_cache.entry(etype).or_insert_with(|| {
                let emt = error_model_types
                    .get(&etype)
                    .expect("error_model_type present for etype referenced by error_model");
                etype_digest(emt)
            });
            ErrorModelFingerprint::new(em.instance(), em.modified_remote_check_models().clone(), digest)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    //! Unit tests covering the cache-key invariant
    //!
    //!   H = H(R, θ, κ, K)
    //!
    //! where R = `relative_program`, θ = resolved `ErrorModelType` +
    //! probability modifier, κ = modified remote check-models / `check_bias`,
    //! K = `committing_local_cids` (commit region for window decoding).
    //!
    //! Each test isolates one axis and asserts that perturbing it changes
    //! the corresponding piece of the key.  These tests would fail under
    //! the previous cache key that only keyed on `RelativeProgram` /
    //! global `eid` values.
    use super::*;
    use crate::bin::ProbabilityModifier;
    use crate::bin::error_model::ErrorModelModifier;
    use crate::bin::error_model_type::{Error, RemoteCheck, RemoteCheckModel, remote_check_model};
    use crate::controller::jit_controller::ErrorModelTypeKey;

    // ─── small constructor helpers ───────────────────────────────────────

    fn pm(probabilities: Vec<f64>) -> ProbabilityModifier {
        ProbabilityModifier {
            probabilities,
            sparse_indices: vec![],
            sparse_probabilities: vec![],
        }
    }

    fn pm_sparse(sparse_indices: Vec<u64>, sparse_probabilities: Vec<f64>) -> ProbabilityModifier {
        ProbabilityModifier {
            probabilities: vec![],
            sparse_indices,
            sparse_probabilities,
        }
    }

    fn error(probability: f64, check_index: u64) -> Error {
        Error {
            checks: vec![RemoteCheck {
                remote_check_model: None,
                check_index,
            }],
            probability,
            ..Default::default()
        }
    }

    fn emt(etype: u64, errors: Vec<Error>) -> ErrorModelType {
        ErrorModelType {
            etype,
            ctype: 1,
            errors,
            remote_check_models: vec![],
            ..Default::default()
        }
    }

    fn em_with_modifier(eid: u64, etype: u64, modifier: Option<ProbabilityModifier>) -> ErrorModel {
        ErrorModel {
            eid,
            etype,
            cid: 1,
            modifier: modifier.map(|p| ErrorModelModifier {
                probability_modifier: Some(p),
                reroute_remote_check_models: vec![],
            }),
            ..Default::default()
        }
    }

    fn remote_check(check_bias: u64, output: u64) -> RemoteCheckModel {
        RemoteCheckModel {
            previous_remote_check_model: None,
            port: Some(remote_check_model::Port::Output(output)),
            expecting_ctype: 0,
            check_bias,
            absolute_cid: None,
            ..Default::default()
        }
    }

    /// Test convenience: build a fingerprint from a borrowed
    /// remote-check-models slice and an `ErrorModelType`, computing the
    /// digest and wrapping the slice in a fresh `Arc`.  Production code
    /// always has the digest precomputed (memoized by
    /// `build_modifier_fingerprints`) and the `Arc` already in hand, so
    /// it calls [`ErrorModelFingerprint::new`] directly.
    fn fingerprint(instance: &ErrorModel, rcm: &[Option<RemoteCheckModel>], emt: &ErrorModelType) -> ErrorModelFingerprint {
        ErrorModelFingerprint::new(instance, Arc::new(rcm.to_vec()), etype_digest(emt))
    }

    // ─── 1. ProbabilityModifierBits ──────────────────────────────────────

    #[test]
    fn probability_modifier_bits_distinguishes_different_dense_values() {
        let a = ProbabilityModifierBits::from(&pm(vec![0.1, 0.2]));
        let b = ProbabilityModifierBits::from(&pm(vec![0.1, 0.3]));
        assert_ne!(a, b);
    }

    #[test]
    fn probability_modifier_bits_distinguishes_different_sparse_values() {
        let a = ProbabilityModifierBits::from(&pm_sparse(vec![0, 2], vec![0.1, 0.4]));
        let b = ProbabilityModifierBits::from(&pm_sparse(vec![0, 2], vec![0.1, 0.5]));
        assert_ne!(a, b);
    }

    #[test]
    fn probability_modifier_bits_distinguishes_different_sparse_indices() {
        let a = ProbabilityModifierBits::from(&pm_sparse(vec![0, 2], vec![0.1, 0.2]));
        let b = ProbabilityModifierBits::from(&pm_sparse(vec![0, 3], vec![0.1, 0.2]));
        assert_ne!(a, b);
    }

    /// `+0.0` and `-0.0` compare equal under `==`, but `to_bits()` preserves
    /// their distinct bit patterns.  Distinguishing them matters because the
    /// downstream hypergraph stores raw probabilities and any user that sets
    /// a negative-zero modifier must not share a cache slot with one that
    /// sets positive-zero.
    #[test]
    fn probability_modifier_bits_distinguishes_positive_and_negative_zero() {
        assert_eq!(0.0_f64, -0.0_f64);
        let a = ProbabilityModifierBits::from(&pm(vec![0.0]));
        let b = ProbabilityModifierBits::from(&pm(vec![-0.0]));
        assert_ne!(a, b);
    }

    /// Two NaNs with different payloads always compare unequal under `==`
    /// (since `NaN != NaN`).  The point here is that `to_bits()` *preserves*
    /// their identity: distinct payload → distinct `ProbabilityModifierBits`.
    /// A naïve `PartialEq` impl on `f64` would lose this information.
    #[test]
    fn probability_modifier_bits_distinguishes_different_nan_payloads() {
        let nan1 = f64::from_bits(0x7ff8_0000_0000_0001);
        let nan2 = f64::from_bits(0x7ff8_0000_0000_0002);
        assert!(nan1.is_nan() && nan2.is_nan());
        let a = ProbabilityModifierBits::from(&pm(vec![nan1]));
        let b = ProbabilityModifierBits::from(&pm(vec![nan2]));
        assert_ne!(a, b);
    }

    #[test]
    fn probability_modifier_bits_equal_for_identical_inputs() {
        let a = ProbabilityModifierBits::from(&pm(vec![0.1, 0.2]));
        let b = ProbabilityModifierBits::from(&pm(vec![0.1, 0.2]));
        assert_eq!(a, b);
    }

    // ─── 2. ErrorModelFingerprint ────────────────────────────────────────

    #[test]
    fn fingerprint_differs_when_probability_modifier_differs() {
        let etype = emt(1, vec![error(0.1, 0)]);
        let em1 = em_with_modifier(1, 1, Some(pm(vec![0.1])));
        let em2 = em_with_modifier(1, 1, Some(pm(vec![0.2])));
        let f1 = fingerprint(&em1, &[], &etype);
        let f2 = fingerprint(&em2, &[], &etype);
        assert_ne!(f1, f2);
    }

    #[test]
    fn fingerprint_differs_when_modifier_present_vs_absent() {
        let etype = emt(1, vec![error(0.1, 0)]);
        let em_no_mod = em_with_modifier(1, 1, None);
        let em_with_mod = em_with_modifier(1, 1, Some(pm(vec![0.1])));
        let f1 = fingerprint(&em_no_mod, &[], &etype);
        let f2 = fingerprint(&em_with_mod, &[], &etype);
        assert_ne!(f1, f2);
    }

    #[test]
    fn fingerprint_differs_when_check_bias_differs() {
        let etype = emt(1, vec![error(0.1, 0)]);
        let em = em_with_modifier(1, 1, None);
        let rcm_a = vec![Some(remote_check(/* check_bias */ 0, /* output port */ 0))];
        let rcm_b = vec![Some(remote_check(/* check_bias */ 5, /* output port */ 0))];
        let f1 = fingerprint(&em, &rcm_a, &etype);
        let f2 = fingerprint(&em, &rcm_b, &etype);
        assert_ne!(f1, f2);
    }

    #[test]
    fn fingerprint_differs_when_remote_check_model_present_vs_absent() {
        let etype = emt(1, vec![error(0.1, 0)]);
        let em = em_with_modifier(1, 1, None);
        let rcm_none = vec![None];
        let rcm_some = vec![Some(remote_check(0, 0))];
        let f1 = fingerprint(&em, &rcm_none, &etype);
        let f2 = fingerprint(&em, &rcm_some, &etype);
        assert_ne!(f1, f2);
    }

    /// Two `ErrorModelType`s with the same `etype` ID but different `errors`
    /// must produce different `etype_digest` values, so that reloading a
    /// library that reuses an `etype` for a different shape invalidates the
    /// cache.  Old key kept only the `etype` id and would incorrectly hit
    /// the cache here.
    #[test]
    fn fingerprint_differs_when_same_etype_id_but_different_errors() {
        let etype_v1 = emt(1, vec![error(0.1, 0)]);
        let etype_v2 = emt(1, vec![error(0.1, 0), error(0.2, 1)]);
        let em = em_with_modifier(1, 1, None);
        let f1 = fingerprint(&em, &[], &etype_v1);
        let f2 = fingerprint(&em, &[], &etype_v2);
        assert_ne!(f1, f2);
        assert_ne!(f1.etype_digest, f2.etype_digest);
    }

    #[test]
    fn fingerprint_differs_when_same_etype_id_but_different_probability() {
        let etype_v1 = emt(1, vec![error(0.1, 0)]);
        let etype_v2 = emt(1, vec![error(0.2, 0)]);
        let em = em_with_modifier(1, 1, None);
        let f1 = fingerprint(&em, &[], &etype_v1);
        let f2 = fingerprint(&em, &[], &etype_v2);
        assert_ne!(f1.etype_digest, f2.etype_digest);
    }

    #[test]
    fn fingerprint_equal_for_identical_inputs() {
        let etype = emt(1, vec![error(0.1, 0)]);
        let em = em_with_modifier(1, 1, Some(pm(vec![0.1])));
        let rcm = vec![Some(remote_check(2, 0))];
        let f1 = fingerprint(&em, &rcm, &etype);
        let f2 = fingerprint(&em, &rcm, &etype);
        assert_eq!(f1, f2);
    }

    /// Lock in that `etype_digest` agrees with the canonical structural
    /// hash used by `TypeCache` (via `ErrorModelTypeKey`).  Drift here would
    /// silently desync the cache key from the JIT type-deduplication logic.
    #[test]
    fn etype_digest_matches_error_model_type_key_hash() {
        let et = emt(1, vec![error(0.1, 0), error(0.2, 1)]);
        let digest = etype_digest(&et);
        let mut hasher = DefaultHasher::new();
        ErrorModelTypeKey(et.clone()).hash(&mut hasher);
        assert_eq!(digest, hasher.finish());
    }

    // ─── 3. DecoderCacheKey ──────────────────────────────────────────────

    fn empty_relative_program() -> RelativeProgram {
        RelativeProgram {
            local_gadgets: vec![],
            count_checks: 0,
        }
    }

    fn fp_with_etype_digest(d: u64) -> ErrorModelFingerprint {
        ErrorModelFingerprint {
            etype_digest: d,
            pm: None,
            remote_check_models: Arc::new(vec![]),
        }
    }

    #[test]
    fn cache_key_differs_when_fingerprints_differ() {
        let r = empty_relative_program();
        let k1 = DecoderCacheKey {
            relative_program: r.clone(),
            error_model_fingerprints: vec![fp_with_etype_digest(1)],
            committing_local_cids: vec![],
        };
        let k2 = DecoderCacheKey {
            relative_program: r,
            error_model_fingerprints: vec![fp_with_etype_digest(2)],
            committing_local_cids: vec![],
        };
        assert_ne!(k1, k2);

        // Round-trip through HashMap: looking up k2 in a map that only holds
        // k1 must miss.  We deliberately avoid asserting on raw hasher output
        // since hashing is non-deterministic across runs.
        let mut map: HashMap<DecoderCacheKey, &'static str> = HashMap::new();
        map.insert(k1.clone(), "first");
        assert_eq!(map.get(&k2), None);
        assert_eq!(map.get(&k1), Some(&"first"));
    }

    #[test]
    fn cache_key_differs_when_committing_local_cids_differ() {
        let r = empty_relative_program();
        let k1 = DecoderCacheKey {
            relative_program: r.clone(),
            error_model_fingerprints: vec![],
            committing_local_cids: vec![0, 1],
        };
        let k2 = DecoderCacheKey {
            relative_program: r,
            error_model_fingerprints: vec![],
            committing_local_cids: vec![0, 2],
        };
        assert_ne!(k1, k2);

        let mut map: HashMap<DecoderCacheKey, &'static str> = HashMap::new();
        map.insert(k1.clone(), "first");
        assert_eq!(map.get(&k2), None);
    }

    /// Demonstrates the dimension that the old cache key (`RelativeProgram`
    /// alone) ignored entirely: two windows that share the same
    /// `RelativeProgram` and per-error-model fingerprints but commit
    /// different cid subsets must map to different cache entries, because
    /// `is_in_commit_region` drops boundary-straddling hyperedges at
    /// hypergraph-construction time.
    #[test]
    fn cache_key_differs_when_only_commit_region_differs() {
        let r = empty_relative_program();
        let fp = fp_with_etype_digest(42);
        let k1 = DecoderCacheKey {
            relative_program: r.clone(),
            error_model_fingerprints: vec![fp.clone()],
            committing_local_cids: vec![0, 1, 2],
        };
        let k2 = DecoderCacheKey {
            relative_program: r,
            error_model_fingerprints: vec![fp],
            committing_local_cids: vec![0, 1],
        };
        assert_ne!(k1, k2);
    }

    #[test]
    fn cache_key_committing_local_cids_order_matters() {
        // The `DecoderCacheKey` itself does not canonicalise — sorting is
        // the caller's responsibility (e.g. `committing_local_cids_sorted`
        // in `window_coordinator.rs`).  Lock the contract: differently
        // ordered vectors with the same contents are *not* equal at the
        // key level.
        let r = empty_relative_program();
        let k1 = DecoderCacheKey {
            relative_program: r.clone(),
            error_model_fingerprints: vec![],
            committing_local_cids: vec![0, 1, 2],
        };
        let k2 = DecoderCacheKey {
            relative_program: r,
            error_model_fingerprints: vec![],
            committing_local_cids: vec![2, 1, 0],
        };
        assert_ne!(k1, k2);
    }

    #[test]
    fn cache_key_equal_for_identical_inputs() {
        let r = empty_relative_program();
        let fp = fp_with_etype_digest(7);
        let k1 = DecoderCacheKey {
            relative_program: r.clone(),
            error_model_fingerprints: vec![fp.clone()],
            committing_local_cids: vec![0, 1],
        };
        let k2 = DecoderCacheKey {
            relative_program: r,
            error_model_fingerprints: vec![fp],
            committing_local_cids: vec![0, 1],
        };
        assert_eq!(k1, k2);

        let mut map: HashMap<DecoderCacheKey, &'static str> = HashMap::new();
        map.insert(k1, "first");
        assert_eq!(map.get(&k2), Some(&"first"));
    }

    /// Two fingerprints differing only by `etype_digest` must produce
    /// different keys when placed in the same slot of
    /// `error_model_fingerprints` — guards the per-slot positional matching
    /// that the coordinators rely on.
    #[test]
    fn cache_key_distinguishes_fingerprint_position() {
        let r = empty_relative_program();
        let fp_a = fp_with_etype_digest(1);
        let fp_b = fp_with_etype_digest(2);
        let k1 = DecoderCacheKey {
            relative_program: r.clone(),
            error_model_fingerprints: vec![fp_a.clone(), fp_b.clone()],
            committing_local_cids: vec![],
        };
        let k2 = DecoderCacheKey {
            relative_program: r,
            error_model_fingerprints: vec![fp_b, fp_a],
            committing_local_cids: vec![],
        };
        assert_ne!(k1, k2);
    }
}
