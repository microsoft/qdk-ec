"""
Benchmarks for stabilizer simulation, comparing paulimer simulators against stim.

This file contains:
1. Per-method benchmarks for paulimer simulators only (OutcomeCompleteSimulation,
   OutcomeFreeSimulation, OutcomeSpecificSimulation)
2. Bulk performance comparisons against stim for canonical circuits
"""

try:
    import stim
    HAS_STIM = True
except ImportError:
    HAS_STIM = False

from paulimer import (
    CliffordUnitary,
    SparsePauli,
    UnitaryOpcode,
    OutcomeCompleteSimulation,
    OutcomeFreeSimulation,
    OutcomeSpecificSimulation,
)


SIMULATION_CLASSES = [
    OutcomeCompleteSimulation,
    OutcomeFreeSimulation,
    OutcomeSpecificSimulation,
]


def pauli_pattern(num_qubits, offset=0):
    """Generate a deterministic Pauli pattern string like 'IXYZIXYZ...'."""
    return "".join(["IXYZ"[(offset + j) % 4] for j in range(num_qubits)])


class SimulationInitialization:
    """Benchmark initialization of simulators."""
    params = [
        [10, 100, 1000],
        ["OutcomeCompleteSimulation", "OutcomeFreeSimulation", "OutcomeSpecificSimulation"],
    ]
    param_names = ["num_qubits", "simulator"]

    def setup(self, num_qubits, simulator):
        self.sim_class = {
            "OutcomeCompleteSimulation": OutcomeCompleteSimulation,
            "OutcomeFreeSimulation": OutcomeFreeSimulation,
            "OutcomeSpecificSimulation": OutcomeSpecificSimulation,
        }[simulator]

    def time_new(self, num_qubits, simulator):
        self.sim_class(num_qubits)

    def time_with_capacity(self, num_qubits, simulator):
        self.sim_class.with_capacity(num_qubits, 100, 50)


class SimulationUnitaryOperations:
    """Benchmark unitary gate operations."""
    params = [
        [10, 100, 1000],
        ["OutcomeCompleteSimulation", "OutcomeFreeSimulation", "OutcomeSpecificSimulation"],
    ]
    param_names = ["num_qubits", "simulator"]

    def setup(self, num_qubits, simulator):
        sim_class = {
            "OutcomeCompleteSimulation": OutcomeCompleteSimulation,
            "OutcomeFreeSimulation": OutcomeFreeSimulation,
            "OutcomeSpecificSimulation": OutcomeSpecificSimulation,
        }[simulator]
        self.sim = sim_class(num_qubits)
        self.num_qubits = num_qubits

    def time_apply_hadamard(self, num_qubits, simulator):
        self.sim.apply_unitary(UnitaryOpcode.Hadamard, [num_qubits // 2])

    def time_apply_cnot(self, num_qubits, simulator):
        self.sim.apply_unitary(UnitaryOpcode.ControlledX, [0, 1])

    def time_apply_cz(self, num_qubits, simulator):
        self.sim.apply_unitary(UnitaryOpcode.ControlledZ, [0, 1])

    def time_apply_sqrt_z(self, num_qubits, simulator):
        self.sim.apply_unitary(UnitaryOpcode.SqrtZ, [num_qubits // 2])

    def time_apply_swap(self, num_qubits, simulator):
        self.sim.apply_unitary(UnitaryOpcode.Swap, [0, 1])


class SimulationPauliOperations:
    """Benchmark Pauli-related operations."""
    params = [
        [10, 100, 1000],
        ["OutcomeCompleteSimulation", "OutcomeFreeSimulation", "OutcomeSpecificSimulation"],
    ]
    param_names = ["num_qubits", "simulator"]

    def setup(self, num_qubits, simulator):
        sim_class = {
            "OutcomeCompleteSimulation": OutcomeCompleteSimulation,
            "OutcomeFreeSimulation": OutcomeFreeSimulation,
            "OutcomeSpecificSimulation": OutcomeSpecificSimulation,
        }[simulator]
        self.sim = sim_class(num_qubits)
        self.num_qubits = num_qubits
        self.sparse_pauli = SparsePauli(pauli_pattern(min(num_qubits, 10)))
        self.single_x = SparsePauli.x(num_qubits // 2)
        self.single_z = SparsePauli.z(0)

    def time_apply_pauli(self, num_qubits, simulator):
        self.sim.apply_pauli(self.sparse_pauli)

    def time_apply_pauli_exp(self, num_qubits, simulator):
        self.sim.apply_pauli_exp(self.sparse_pauli)

    def time_apply_controlled_pauli(self, num_qubits, simulator):
        self.sim.apply_pauli(self.single_x, controlled_by=self.single_z)


class SimulationCliffordOperations:
    """Benchmark Clifford and permutation operations."""
    params = [
        [10, 100, 1000],
        ["OutcomeCompleteSimulation", "OutcomeFreeSimulation", "OutcomeSpecificSimulation"],
    ]
    param_names = ["num_qubits", "simulator"]

    def setup(self, num_qubits, simulator):
        sim_class = {
            "OutcomeCompleteSimulation": OutcomeCompleteSimulation,
            "OutcomeFreeSimulation": OutcomeFreeSimulation,
            "OutcomeSpecificSimulation": OutcomeSpecificSimulation,
        }[simulator]
        self.sim = sim_class(num_qubits)
        self.num_qubits = num_qubits
        self.hadamard_clifford = CliffordUnitary.from_name("Hadamard", [0], 1)
        self.cnot_clifford = CliffordUnitary.from_name("ControlledX", [0, 1], 2)
        self.small_permutation = [1, 0]

    def time_apply_clifford_1q(self, num_qubits, simulator):
        self.sim.apply_clifford(self.hadamard_clifford, supported_by=[num_qubits // 2])

    def time_apply_clifford_2q(self, num_qubits, simulator):
        self.sim.apply_clifford(self.cnot_clifford, supported_by=[0, 1])

    def time_apply_permutation(self, num_qubits, simulator):
        self.sim.apply_permutation(self.small_permutation, supported_by=[0, 1])


class SimulationMeasurement:
    """Benchmark measurement operations."""
    params = [
        [10, 100, 1000],
        ["OutcomeCompleteSimulation", "OutcomeFreeSimulation", "OutcomeSpecificSimulation"],
    ]
    param_names = ["num_qubits", "simulator"]

    def setup(self, num_qubits, simulator):
        sim_class = {
            "OutcomeCompleteSimulation": OutcomeCompleteSimulation,
            "OutcomeFreeSimulation": OutcomeFreeSimulation,
            "OutcomeSpecificSimulation": OutcomeSpecificSimulation,
        }[simulator]
        self.sim_class = sim_class
        self.num_qubits = num_qubits
        self.single_z = SparsePauli.z(0)
        self.multi_qubit_observable = SparsePauli("ZZZ")

    def time_measure_single_qubit(self, num_qubits, simulator):
        sim = self.sim_class(num_qubits)
        sim.measure(self.single_z)

    def time_measure_multi_qubit(self, num_qubits, simulator):
        sim = self.sim_class(num_qubits)
        sim.measure(self.multi_qubit_observable)

    def time_is_stabilizer(self, num_qubits, simulator):
        sim = self.sim_class(num_qubits)
        sim.is_stabilizer(self.single_z)


class SimulationConditionalOperations:
    """Benchmark conditional and outcome-dependent operations."""
    params = [
        [10, 100, 1000],
        ["OutcomeCompleteSimulation", "OutcomeFreeSimulation", "OutcomeSpecificSimulation"],
    ]
    param_names = ["num_qubits", "simulator"]

    def setup(self, num_qubits, simulator):
        sim_class = {
            "OutcomeCompleteSimulation": OutcomeCompleteSimulation,
            "OutcomeFreeSimulation": OutcomeFreeSimulation,
            "OutcomeSpecificSimulation": OutcomeSpecificSimulation,
        }[simulator]
        self.sim = sim_class(num_qubits)
        self.sim.measure(SparsePauli.z(0))
        self.num_qubits = num_qubits
        self.correction_pauli = SparsePauli.x(1)

    def time_apply_conditional_pauli(self, num_qubits, simulator):
        self.sim.apply_conditional_pauli(self.correction_pauli, outcomes=[0], parity=False)


# =============================================================================
# Bulk Performance Comparisons against Stim
# =============================================================================

class SimulationBulkRandomCliffordCircuit:
    """
    Benchmark random Clifford circuit simulation against stim.
    
    Simulates a circuit with H gates on all qubits, followed by a layer of 
    CNOTs, and then measurements.
    """
    params = [[10, 100, 200]]
    param_names = ["num_qubits"]
    
    number = 1
    repeat = 3
    warmup_time = 0.1

    def setup(self, num_qubits):
        self.num_qubits = num_qubits
        if HAS_STIM:
            circuit_str_parts = []
            for i in range(num_qubits):
                circuit_str_parts.append(f"H {i}")
            for i in range(num_qubits - 1):
                circuit_str_parts.append(f"CNOT {i} {i+1}")
            for i in range(num_qubits):
                circuit_str_parts.append(f"M {i}")
            self.stim_circuit = stim.Circuit("\n".join(circuit_str_parts))

    def time_paulimer_outcome_complete(self, num_qubits):
        for _ in range(100):
            sim = OutcomeCompleteSimulation.with_capacity(num_qubits, num_qubits, num_qubits)
            for i in range(num_qubits):
                sim.apply_unitary(UnitaryOpcode.Hadamard, [i])
            for i in range(num_qubits - 1):
                sim.apply_unitary(UnitaryOpcode.ControlledX, [i, i + 1])
            for i in range(num_qubits):
                sim.measure(SparsePauli.z(i))

    def time_paulimer_outcome_free(self, num_qubits):
        for _ in range(100):
            sim = OutcomeFreeSimulation(num_qubits)
            for i in range(num_qubits):
                sim.apply_unitary(UnitaryOpcode.Hadamard, [i])
            for i in range(num_qubits - 1):
                sim.apply_unitary(UnitaryOpcode.ControlledX, [i, i + 1])
            for i in range(num_qubits):
                sim.measure(SparsePauli.z(i))

    def time_paulimer_outcome_specific(self, num_qubits):
        for _ in range(100):
            sim = OutcomeSpecificSimulation(num_qubits)
            for i in range(num_qubits):
                sim.apply_unitary(UnitaryOpcode.Hadamard, [i])
            for i in range(num_qubits - 1):
                sim.apply_unitary(UnitaryOpcode.ControlledX, [i, i + 1])
            for i in range(num_qubits):
                sim.measure(SparsePauli.z(i))

    if HAS_STIM:
        def time_stim_tableau_simulator(self, num_qubits):
            for _ in range(100):
                sim = stim.TableauSimulator()
                for i in range(num_qubits):
                    sim.h(i)
                for i in range(num_qubits - 1):
                    sim.cnot(i, i + 1)
                for i in range(num_qubits):
                    sim.measure(i)

        def time_stim_circuit_sample(self, num_qubits):
            self.stim_circuit.compile_sampler().sample(100)


class SimulationBulkGHZStatePreparation:
    """
    Benchmark GHZ state preparation and measurement against stim.
    
    Prepares |GHZ> = (|00...0> + |11...1>) / sqrt(2) and measures all qubits.
    """
    params = [[10, 100, 1000]]
    param_names = ["num_qubits"]
    
    number = 1
    repeat = 3
    warmup_time = 0.1

    def setup(self, num_qubits):
        self.num_qubits = num_qubits
        if HAS_STIM:
            circuit_str_parts = ["H 0"]
            for i in range(num_qubits - 1):
                circuit_str_parts.append(f"CNOT 0 {i+1}")
            for i in range(num_qubits):
                circuit_str_parts.append(f"M {i}")
            self.stim_circuit = stim.Circuit("\n".join(circuit_str_parts))

    def time_paulimer_outcome_complete(self, num_qubits):
        for _ in range(100):
            sim = OutcomeCompleteSimulation.with_capacity(num_qubits, num_qubits, num_qubits)
            sim.apply_unitary(UnitaryOpcode.Hadamard, [0])
            for i in range(num_qubits - 1):
                sim.apply_unitary(UnitaryOpcode.ControlledX, [0, i + 1])
            for i in range(num_qubits):
                sim.measure(SparsePauli.z(i))

    def time_paulimer_outcome_free(self, num_qubits):
        for _ in range(100):
            sim = OutcomeFreeSimulation(num_qubits)
            sim.apply_unitary(UnitaryOpcode.Hadamard, [0])
            for i in range(num_qubits - 1):
                sim.apply_unitary(UnitaryOpcode.ControlledX, [0, i + 1])
            for i in range(num_qubits):
                sim.measure(SparsePauli.z(i))

    def time_paulimer_outcome_specific(self, num_qubits):
        for _ in range(100):
            sim = OutcomeSpecificSimulation(num_qubits)
            sim.apply_unitary(UnitaryOpcode.Hadamard, [0])
            for i in range(num_qubits - 1):
                sim.apply_unitary(UnitaryOpcode.ControlledX, [0, i + 1])
            for i in range(num_qubits):
                sim.measure(SparsePauli.z(i))

    if HAS_STIM:
        def time_stim_tableau_simulator(self, num_qubits):
            for _ in range(100):
                sim = stim.TableauSimulator()
                sim.h(0)
                for i in range(num_qubits - 1):
                    sim.cnot(0, i + 1)
                for i in range(num_qubits):
                    sim.measure(i)

        def time_stim_circuit_sample(self, num_qubits):
            self.stim_circuit.compile_sampler().sample(100)


class SimulationBulkQFTLikeCircuit:
    """
    Benchmark QFT-like Clifford circuit (Hadamards + controlled-Z ladder).
    
    This approximates the structure of a QFT with only Clifford gates.
    Note: O(n^2) gates, so 1000 qubits would be ~500K gates.
    """
    params = [[10, 100]]
    param_names = ["num_qubits"]
    
    number = 1
    repeat = 3
    warmup_time = 0.1

    def setup(self, num_qubits):
        self.num_qubits = num_qubits
        if HAS_STIM:
            circuit_str_parts = []
            for i in range(num_qubits):
                circuit_str_parts.append(f"H {i}")
                for j in range(i + 1, num_qubits):
                    circuit_str_parts.append(f"CZ {i} {j}")
            for i in range(num_qubits):
                circuit_str_parts.append(f"M {i}")
            self.stim_circuit = stim.Circuit("\n".join(circuit_str_parts))

    def time_paulimer_outcome_complete(self, num_qubits):
        for _ in range(100):
            sim = OutcomeCompleteSimulation.with_capacity(num_qubits, num_qubits, num_qubits)
            for i in range(num_qubits):
                sim.apply_unitary(UnitaryOpcode.Hadamard, [i])
                for j in range(i + 1, num_qubits):
                    sim.apply_unitary(UnitaryOpcode.ControlledZ, [i, j])
            for i in range(num_qubits):
                sim.measure(SparsePauli.z(i))

    def time_paulimer_outcome_free(self, num_qubits):
        for _ in range(100):
            sim = OutcomeFreeSimulation(num_qubits)
            for i in range(num_qubits):
                sim.apply_unitary(UnitaryOpcode.Hadamard, [i])
                for j in range(i + 1, num_qubits):
                    sim.apply_unitary(UnitaryOpcode.ControlledZ, [i, j])
            for i in range(num_qubits):
                sim.measure(SparsePauli.z(i))

    def time_paulimer_outcome_specific(self, num_qubits):
        for _ in range(100):
            sim = OutcomeSpecificSimulation(num_qubits)
            for i in range(num_qubits):
                sim.apply_unitary(UnitaryOpcode.Hadamard, [i])
                for j in range(i + 1, num_qubits):
                    sim.apply_unitary(UnitaryOpcode.ControlledZ, [i, j])
            for i in range(num_qubits):
                sim.measure(SparsePauli.z(i))

    if HAS_STIM:
        def time_stim_tableau_simulator(self, num_qubits):
            for _ in range(100):
                sim = stim.TableauSimulator()
                for i in range(num_qubits):
                    sim.h(i)
                    for j in range(i + 1, num_qubits):
                        sim.cz(i, j)
                for i in range(num_qubits):
                    sim.measure(i)

        def time_stim_circuit_sample(self, num_qubits):
            self.stim_circuit.compile_sampler().sample(100)


class SimulationBulkSurfaceCodeLikeStabilizers:
    """
    Benchmark stabilizer measurement patterns similar to surface code.
    
    Creates a grid of XXXX and ZZZZ stabilizer measurements.
    """
    params = [[9, 100, 1024]]  # Perfect squares for grid (~3x3, 10x10, 32x32)
    param_names = ["num_qubits"]
    
    number = 1
    repeat = 3
    warmup_time = 0.1

    def setup(self, num_qubits):
        import math
        self.grid_size = int(math.sqrt(num_qubits))
        self.num_qubits = self.grid_size ** 2

        # Build X-type stabilizers (on plaquettes)
        self.x_stabilizers = []
        for row in range(self.grid_size - 1):
            for col in range(self.grid_size - 1):
                indices = [
                    row * self.grid_size + col,
                    row * self.grid_size + col + 1,
                    (row + 1) * self.grid_size + col,
                    (row + 1) * self.grid_size + col + 1,
                ]
                pauli_dict = {idx: "X" for idx in indices}
                self.x_stabilizers.append(SparsePauli(pauli_dict))

        # Build Z-type stabilizers (on vertices - shifted)
        self.z_stabilizers = []
        for row in range(1, self.grid_size):
            for col in range(1, self.grid_size):
                indices = [
                    row * self.grid_size + col,
                    row * self.grid_size + col - 1,
                    (row - 1) * self.grid_size + col,
                    (row - 1) * self.grid_size + col - 1,
                ]
                pauli_dict = {idx: "Z" for idx in indices}
                self.z_stabilizers.append(SparsePauli(pauli_dict))

        if HAS_STIM:
            self._build_stim_circuit()

    def _build_stim_circuit(self):
        circuit_str_parts = []
        for row in range(self.grid_size - 1):
            for col in range(self.grid_size - 1):
                indices = [
                    row * self.grid_size + col,
                    row * self.grid_size + col + 1,
                    (row + 1) * self.grid_size + col,
                    (row + 1) * self.grid_size + col + 1,
                ]
                circuit_str_parts.append(f"MPP X{indices[0]}*X{indices[1]}*X{indices[2]}*X{indices[3]}")
        for row in range(1, self.grid_size):
            for col in range(1, self.grid_size):
                indices = [
                    row * self.grid_size + col,
                    row * self.grid_size + col - 1,
                    (row - 1) * self.grid_size + col,
                    (row - 1) * self.grid_size + col - 1,
                ]
                circuit_str_parts.append(f"MPP Z{indices[0]}*Z{indices[1]}*Z{indices[2]}*Z{indices[3]}")
        self.stim_circuit = stim.Circuit("\n".join(circuit_str_parts))

    def time_paulimer_outcome_complete(self, num_qubits):
        num_stabilizers = len(self.x_stabilizers) + len(self.z_stabilizers)
        for _ in range(100):
            sim = OutcomeCompleteSimulation.with_capacity(self.num_qubits, num_stabilizers, num_stabilizers)
            for stab in self.x_stabilizers:
                sim.measure(stab)
            for stab in self.z_stabilizers:
                sim.measure(stab)

    def time_paulimer_outcome_free(self, num_qubits):
        for _ in range(100):
            sim = OutcomeFreeSimulation(self.num_qubits)
            for stab in self.x_stabilizers:
                sim.measure(stab)
            for stab in self.z_stabilizers:
                sim.measure(stab)

    def time_paulimer_outcome_specific(self, num_qubits):
        for _ in range(100):
            sim = OutcomeSpecificSimulation(self.num_qubits)
            for stab in self.x_stabilizers:
                sim.measure(stab)
            for stab in self.z_stabilizers:
                sim.measure(stab)

    if HAS_STIM:
        def time_stim_circuit_sample(self, num_qubits):
            self.stim_circuit.compile_sampler().sample(100)


class SimulationBulkManyMeasurements:
    """
    Benchmark circuits with many measurements (outcome tracking overhead).
    """
    params = [[10, 100, 1000]]
    param_names = ["num_measurements"]
    
    number = 1
    repeat = 3
    warmup_time = 0.1

    def setup(self, num_measurements):
        self.num_measurements = num_measurements
        self.num_qubits = 10
        if HAS_STIM:
            circuit_str_parts = []
            for i in range(num_measurements):
                qubit = i % self.num_qubits
                circuit_str_parts.append(f"H {qubit}")
                circuit_str_parts.append(f"M {qubit}")
                circuit_str_parts.append(f"R {qubit}")
            self.stim_circuit = stim.Circuit("\n".join(circuit_str_parts))

    def time_paulimer_outcome_complete(self, num_measurements):
        for _ in range(100):
            sim = OutcomeCompleteSimulation.with_capacity(self.num_qubits, num_measurements, num_measurements)
            for i in range(num_measurements):
                qubit = i % self.num_qubits
                sim.apply_unitary(UnitaryOpcode.Hadamard, [qubit])
                sim.measure(SparsePauli.z(qubit))

    def time_paulimer_outcome_free(self, num_measurements):
        for _ in range(100):
            sim = OutcomeFreeSimulation(self.num_qubits)
            for i in range(num_measurements):
                qubit = i % self.num_qubits
                sim.apply_unitary(UnitaryOpcode.Hadamard, [qubit])
                sim.measure(SparsePauli.z(qubit))

    def time_paulimer_outcome_specific(self, num_measurements):
        for _ in range(100):
            sim = OutcomeSpecificSimulation(self.num_qubits)
            for i in range(num_measurements):
                qubit = i % self.num_qubits
                sim.apply_unitary(UnitaryOpcode.Hadamard, [qubit])
                sim.measure(SparsePauli.z(qubit))

    if HAS_STIM:
        def time_stim_tableau_simulator(self, num_measurements):
            for _ in range(100):
                sim = stim.TableauSimulator()
                for i in range(num_measurements):
                    qubit = i % self.num_qubits
                    sim.h(qubit)
                    sim.measure(qubit)
                    sim.reset(qubit)

        def time_stim_circuit_sample(self, num_measurements):
            self.stim_circuit.compile_sampler().sample(100)

