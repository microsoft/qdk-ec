use super::*;
use crate::phased_outcome_complete_simulation::PhasedOutcomeCompleteSimulation;
use crate::{Circuit, CircuitBuilder, QubitId, Simulation};
use paulimer::core::{x, z};
use paulimer::pauli::SparsePauli;
use paulimer::{PositionedPauliObservable, UnitaryOp};
use proptest::prelude::*;
use rand::SeedableRng;
use std::ops::Range;

mod comparison;
use comparison::*;
mod ejection;
mod preparation;
use ejection::*;
mod explicit;
use explicit::*;
