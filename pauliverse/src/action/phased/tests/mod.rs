use super::*;
use crate::phased_outcome_complete_simulation::PhasedOutcomeCompleteSimulation;
use crate::{Circuit, CircuitBuilder, QubitId, Simulation};
use binar::BitwiseMut;
use paulimer::core::{x, z};
use paulimer::pauli::SparsePauli;
use paulimer::{PositionedPauliObservable, UnitaryOp};
use proptest::prelude::*;
use rand::SeedableRng;
use std::ops::Range;

mod comparison;
use comparison::*;
mod preparation;
use preparation::*;
mod ejection;
use ejection::*;
mod explicit;
use explicit::*;
