//! Python bindings for noise types: `PauliFault`, `PauliDistribution`, `OutcomeCondition`.

#![allow(clippy::must_use_candidate)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::needless_pass_by_value)]

use crate::PySparsePauli;
use pauliverse::noise::{OutcomeCondition, PauliDistribution, PauliFault};
use pyo3::prelude::*;

/// Python wrapper for OutcomeCondition.
#[derive(Clone, Debug)]
#[pyclass(name = "OutcomeCondition")]
pub struct PyOutcomeCondition {
    pub inner: OutcomeCondition,
}

#[pymethods]
impl PyOutcomeCondition {
    /// Create a condition that triggers when the XOR of outcomes equals parity.
    #[new]
    #[pyo3(signature = (outcomes, parity=true))]
    pub fn new(outcomes: Vec<usize>, parity: bool) -> Self {
        Self {
            inner: OutcomeCondition::new(&outcomes, parity),
        }
    }

    /// The outcome IDs to check.
    #[getter]
    pub fn outcomes(&self) -> Vec<usize> {
        self.inner.outcomes.to_vec()
    }

    /// The required parity of XOR of the outcomes.
    #[getter]
    pub fn parity(&self) -> bool {
        self.inner.parity
    }

    fn __repr__(&self) -> String {
        format!(
            "OutcomeCondition(outcomes={:?}, parity={})",
            self.inner.outcomes.to_vec(),
            self.inner.parity
        )
    }
}

/// Python wrapper for PauliDistribution.
#[derive(Clone, Debug)]
#[pyclass(name = "PauliDistribution")]
pub struct PyPauliDistribution {
    pub inner: PauliDistribution,
}

#[pymethods]
impl PyPauliDistribution {
    /// Create a depolarizing distribution on the given qubits.
    /// Samples uniformly from all non-identity Paulis on these qubits.
    #[staticmethod]
    pub fn depolarizing(qubits: Vec<usize>) -> Self {
        Self {
            inner: PauliDistribution::depolarizing(&qubits),
        }
    }

    /// Create a single deterministic Pauli.
    #[staticmethod]
    pub fn single(pauli: &PySparsePauli) -> Self {
        Self {
            inner: PauliDistribution::single(pauli.inner.clone()),
        }
    }

    /// Create a uniform distribution over an explicit list of Paulis.
    #[staticmethod]
    pub fn uniform(paulis: Vec<PySparsePauli>) -> Self {
        Self {
            inner: PauliDistribution::uniform(paulis.into_iter().map(|p| p.inner).collect()),
        }
    }

    /// Create a weighted distribution from (Pauli, weight) pairs.
    /// Weights are normalized to sum to 1.
    #[staticmethod]
    pub fn weighted(pairs: Vec<(PySparsePauli, f64)>) -> Self {
        Self {
            inner: PauliDistribution::weighted(pairs.into_iter().map(|(p, w)| (p.inner, w)).collect()),
        }
    }

    /// Returns all elements as (SparsePauli, probability) pairs.
    #[getter]
    pub fn elements(&self) -> Vec<(PySparsePauli, f64)> {
        self.inner
            .elements()
            .into_iter()
            .map(|(p, prob)| (PySparsePauli { inner: p }, prob))
            .collect()
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            PauliDistribution::Single(p) => format!("PauliDistribution.single({p:?})"),
            PauliDistribution::DepolarizingOnQubits(qubits) => {
                format!("PauliDistribution.depolarizing({:?})", qubits.to_vec())
            }
            PauliDistribution::UniformOver(paulis) => {
                format!("PauliDistribution.uniform([{} Paulis])", paulis.len())
            }
            PauliDistribution::Weighted { paulis, .. } => {
                format!("PauliDistribution.weighted([{} Paulis])", paulis.len())
            }
        }
    }
}

/// Python wrapper for PauliFault.
#[derive(Clone, Debug)]
#[pyclass(name = "PauliFault")]
pub struct PyFault {
    pub inner: PauliFault,
}

#[pymethods]
impl PyFault {
    /// Create a simple depolarizing noise on the given qubits.
    #[staticmethod]
    pub fn depolarizing(probability: f64, qubits: Vec<usize>) -> Self {
        Self {
            inner: PauliFault::depolarizing(&qubits, probability),
        }
    }

    /// Create a noise instruction with the given distribution and probability.
    #[new]
    #[pyo3(signature = (probability, distribution, correlation_id=None, condition=None))]
    pub fn new(
        probability: f64,
        distribution: &PyPauliDistribution,
        correlation_id: Option<u64>,
        condition: Option<&PyOutcomeCondition>,
    ) -> Self {
        Self {
            inner: PauliFault {
                probability,
                distribution: distribution.inner.clone(),
                correlation_id,
                condition: condition.map(|c| c.inner.clone()),
            },
        }
    }

    /// The probability that a fault occurs.
    #[getter]
    pub fn probability(&self) -> f64 {
        self.inner.probability
    }

    /// The Pauli distribution for this fault.
    #[getter]
    pub fn distribution(&self) -> PyPauliDistribution {
        PyPauliDistribution {
            inner: self.inner.distribution.clone(),
        }
    }

    /// The correlation ID for correlated faults (None if not correlated).
    #[getter]
    pub fn correlation_id(&self) -> Option<u64> {
        self.inner.correlation_id
    }

    /// The condition for this fault (None if unconditional).
    #[getter]
    pub fn condition(&self) -> Option<PyOutcomeCondition> {
        self.inner.condition.clone().map(|c| PyOutcomeCondition { inner: c })
    }

    fn __repr__(&self) -> String {
        let mut parts = vec![format!("probability={}", self.inner.probability)];
        if let Some(id) = self.inner.correlation_id {
            parts.push(format!("correlation_id={id}"));
        }
        if self.inner.condition.is_some() {
            parts.push("conditional=True".to_string());
        }
        format!("PauliFault({})", parts.join(", "))
    }
}
