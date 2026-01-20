"""
Benchmarks for CliffordUnitary operations, comparing against Stim's Tableau and Qiskit's Clifford.
"""

import pickle

try:
    import stim
    HAS_STIM = True
except ImportError:
    HAS_STIM = False

try:
    from qiskit.quantum_info import Clifford as QiskitClifford, Pauli as QiskitPauli
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

from paulimer import CliffordUnitary, DensePauli, SparsePauli, UnitaryOpcode, split_qubit_cliffords_and_css, split_phased_css, is_diagonal_resource_encoder


def build_clifford_paulimer(num_qubits, variant=1):
    """Build a deterministic non-trivial Clifford for paulimer.
    
    variant=1: H on all qubits, CX ladder
    variant=2: S on all qubits, CZ ladder
    """
    c = CliffordUnitary.identity(num_qubits)
    if variant == 1:
        for i in range(num_qubits):
            c.left_mul(UnitaryOpcode.Hadamard, [i])
        for i in range(num_qubits - 1):
            c.left_mul(UnitaryOpcode.ControlledX, [i, i + 1])
    else:
        for i in range(num_qubits):
            c.left_mul(UnitaryOpcode.SqrtZ, [i])
        for i in range(num_qubits - 1):
            c.left_mul(UnitaryOpcode.ControlledZ, [i, i + 1])
    return c


def build_clifford_stim(num_qubits, variant=1):
    """Build a deterministic non-trivial Clifford for stim (matching paulimer)."""
    if not HAS_STIM:
        return None
    t = stim.Tableau(num_qubits)
    h_gate = stim.Tableau.from_named_gate("H")
    s_gate = stim.Tableau.from_named_gate("S")
    cnot_gate = stim.Tableau.from_named_gate("CNOT")
    cz_gate = stim.Tableau.from_named_gate("CZ")
    if variant == 1:
        for i in range(num_qubits):
            t.append(h_gate, [i])
        for i in range(num_qubits - 1):
            t.append(cnot_gate, [i, i + 1])
    else:
        for i in range(num_qubits):
            t.append(s_gate, [i])
        for i in range(num_qubits - 1):
            t.append(cz_gate, [i, i + 1])
    return t


def build_clifford_qiskit(num_qubits, variant=1):
    """Build a deterministic non-trivial Clifford for qiskit (matching paulimer)."""
    if not HAS_QISKIT:
        return None
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(num_qubits)
    if variant == 1:
        for i in range(num_qubits):
            qc.h(i)
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
    else:
        for i in range(num_qubits):
            qc.s(i)
        for i in range(num_qubits - 1):
            qc.cz(i, i + 1)
    return QiskitClifford.from_circuit(qc)


def pauli_pattern(num_qubits, offset=0):
    """Generate a deterministic Pauli pattern string like 'IXYZIXYZ...'."""
    return "".join(["IXYZ"[(offset + j) % 4] for j in range(num_qubits)])


if HAS_STIM:
    STIM_H = stim.Tableau.from_named_gate("H")
    STIM_S = stim.Tableau.from_named_gate("S")
    STIM_CNOT = stim.Tableau.from_named_gate("CNOT")
    STIM_CZ = stim.Tableau.from_named_gate("CZ")


class CliffordUnitaryInitialization:
    """Benchmark initialization of Clifford unitaries."""
    params = [[10, 100]]
    param_names = ['num_qubits']
    
    def setup(self, num_qubits):
        c_for_string = build_clifford_paulimer(num_qubits, variant=1)
        self.clifford_string = str(c_for_string)
        self.preimages_paulimer = []
        for i in range(num_qubits):
            self.preimages_paulimer.append(DensePauli.x(i, num_qubits))
            self.preimages_paulimer.append(DensePauli.z(i, num_qubits))
        if HAS_STIM:
            self.preimages_stim_xs = []
            self.preimages_stim_zs = []
            for i in range(num_qubits):
                self.preimages_stim_xs.append(stim.PauliString("I" * i + "X" + "I" * (num_qubits - i - 1)))
                self.preimages_stim_zs.append(stim.PauliString("I" * i + "Z" + "I" * (num_qubits - i - 1)))
        if HAS_QISKIT:
            from qiskit import QuantumCircuit
            self.circuit_qiskit = QuantumCircuit(num_qubits)
            for i in range(num_qubits):
                self.circuit_qiskit.h(i)
            self.identity_circuit_qiskit = QuantumCircuit(num_qubits)
    
    def time_identity_paulimer(self, num_qubits):
        CliffordUnitary.identity(num_qubits)
    
    def time_from_string_paulimer(self, num_qubits):
        CliffordUnitary.from_string(self.clifford_string)
    
    def time_from_preimages_paulimer(self, num_qubits):
        CliffordUnitary.from_preimages(self.preimages_paulimer)
    
    if HAS_STIM:
        def time_identity_stim(self, num_qubits):
            stim.Tableau(num_qubits)
        
        def time_from_named_gate_stim(self, num_qubits):
            stim.Tableau.from_named_gate("H")
        
        def time_from_conjugated_generators_stim(self, num_qubits):
            stim.Tableau.from_conjugated_generators(
                xs=self.preimages_stim_xs,
                zs=self.preimages_stim_zs
            )
    
    if HAS_QISKIT:
        def time_identity_qiskit(self, num_qubits):
            QiskitClifford(self.identity_circuit_qiskit)
        
        def time_from_circuit_qiskit(self, num_qubits):
            QiskitClifford.from_circuit(self.circuit_qiskit)


class CliffordUnitaryComposition:
    """Benchmark composition (multiplication) of Clifford unitaries."""
    params = [[10, 100]]
    param_names = ['num_qubits']
    
    def setup(self, num_qubits):
        self.c1_paulimer = build_clifford_paulimer(num_qubits, variant=1)
        self.c2_paulimer = build_clifford_paulimer(num_qubits, variant=2)
        if HAS_STIM:
            self.t1_stim = build_clifford_stim(num_qubits, variant=1)
            self.t2_stim = build_clifford_stim(num_qubits, variant=2)
        if HAS_QISKIT:
            self.c1_qiskit = build_clifford_qiskit(num_qubits, variant=1)
            self.c2_qiskit = build_clifford_qiskit(num_qubits, variant=2)
    
    def time_multiply_paulimer(self, num_qubits):
        _ = self.c1_paulimer * self.c2_paulimer
    
    if HAS_STIM:
        def time_multiply_stim(self, num_qubits):
            _ = self.t1_stim * self.t2_stim
        
        def time_then_stim(self, num_qubits):
            _ = self.t1_stim.then(self.t2_stim)
    
    if HAS_QISKIT:
        def time_compose_qiskit(self, num_qubits):
            _ = self.c1_qiskit.compose(self.c2_qiskit)
        
        def time_dot_qiskit(self, num_qubits):
            _ = self.c1_qiskit.dot(self.c2_qiskit)


class CliffordUnitaryInverse:
    """Benchmark inverse computation."""
    params = [[10, 100]]
    param_names = ['num_qubits']
    
    def setup(self, num_qubits):
        self.c_paulimer = build_clifford_paulimer(num_qubits, variant=1)
        if HAS_STIM:
            self.t_stim = build_clifford_stim(num_qubits, variant=1)
        if HAS_QISKIT:
            self.c_qiskit = build_clifford_qiskit(num_qubits, variant=1)
    
    def time_inverse_paulimer(self, num_qubits):
        _ = self.c_paulimer.inverse()
    
    if HAS_STIM:
        def time_inverse_stim(self, num_qubits):
            _ = self.t_stim.inverse()
        
        def time_inverse_unsigned_stim(self, num_qubits):
            _ = self.t_stim.inverse(unsigned=True)
    
    if HAS_QISKIT:
        def time_adjoint_qiskit(self, num_qubits):
            _ = self.c_qiskit.adjoint()


class CliffordUnitaryConjugation:
    """Benchmark Pauli conjugation (image/preimage computation)."""
    params = [[10, 100]]
    param_names = ['num_qubits']
    
    def setup(self, num_qubits):
        self.c_paulimer = build_clifford_paulimer(num_qubits, variant=1)
        pattern = pauli_pattern(num_qubits)
        self.pauli_sparse_paulimer = SparsePauli(pattern)
        if HAS_STIM:
            self.t_stim = build_clifford_stim(num_qubits, variant=1)
            self.pauli_stim = stim.PauliString(pattern)
        if HAS_QISKIT:
            self.c_qiskit = build_clifford_qiskit(num_qubits, variant=1)
            self.pauli_qiskit = QiskitPauli(pattern[::-1])
    
    def time_image_x_paulimer(self, num_qubits):
        _ = self.c_paulimer.image_x(0)
    
    def time_image_z_paulimer(self, num_qubits):
        _ = self.c_paulimer.image_z(0)
    
    def time_preimage_x_paulimer(self, num_qubits):
        _ = self.c_paulimer.preimage_x(0)
    
    def time_preimage_z_paulimer(self, num_qubits):
        _ = self.c_paulimer.preimage_z(0)
    
    def time_image_sparse_paulimer(self, num_qubits):
        _ = self.c_paulimer.image_of(self.pauli_sparse_paulimer)
    
    def time_preimage_sparse_paulimer(self, num_qubits):
        _ = self.c_paulimer.preimage_of(self.pauli_sparse_paulimer)
    
    if HAS_STIM:
        def time_x_output_stim(self, num_qubits):
            _ = self.t_stim.x_output(0)
        
        def time_z_output_stim(self, num_qubits):
            _ = self.t_stim.z_output(0)
        
        def time_inverse_x_output_stim(self, num_qubits):
            _ = self.t_stim.inverse_x_output(0)
        
        def time_inverse_z_output_stim(self, num_qubits):
            _ = self.t_stim.inverse_z_output(0)
        
        def time_call_stim(self, num_qubits):
            _ = self.t_stim(self.pauli_stim)
    
    if HAS_QISKIT:
        def time_evolve_qiskit(self, num_qubits):
            _ = self.pauli_qiskit.evolve(self.c_qiskit)


class CliffordUnitaryLeftMultiplication:
    """Benchmark in-place left multiplication operations."""
    params = [[10, 50]]
    param_names = ['num_qubits']
    
    def setup(self, num_qubits):
        self.num_qubits = num_qubits
        self.c_paulimer = CliffordUnitary.identity(num_qubits)
        self.pauli_paulimer = SparsePauli("X" * num_qubits)
        if HAS_STIM:
            self.t_stim = stim.Tableau(num_qubits)
    
    def time_left_mul_h_paulimer(self, num_qubits):
        for i in range(self.num_qubits):
            self.c_paulimer.left_mul(UnitaryOpcode.Hadamard, [i])
    
    def time_left_mul_s_paulimer(self, num_qubits):
        for i in range(self.num_qubits):
            self.c_paulimer.left_mul(UnitaryOpcode.SqrtZ, [i])
    
    def time_left_mul_cx_paulimer(self, num_qubits):
        for i in range(self.num_qubits - 1):
            self.c_paulimer.left_mul(UnitaryOpcode.ControlledX, [i, i + 1])
    
    def time_left_mul_cz_paulimer(self, num_qubits):
        for i in range(self.num_qubits - 1):
            self.c_paulimer.left_mul(UnitaryOpcode.ControlledZ, [i, i + 1])
    
    def time_left_mul_pauli_paulimer(self, num_qubits):
        self.c_paulimer.left_mul_pauli(self.pauli_paulimer)
    
    def time_left_mul_x_paulimer(self, num_qubits):
        for i in range(self.num_qubits):
            self.c_paulimer.left_mul(UnitaryOpcode.X, [i])
    
    def time_left_mul_y_paulimer(self, num_qubits):
        for i in range(self.num_qubits):
            self.c_paulimer.left_mul(UnitaryOpcode.Y, [i])
    
    def time_left_mul_z_paulimer(self, num_qubits):
        for i in range(self.num_qubits):
            self.c_paulimer.left_mul(UnitaryOpcode.Z, [i])
    
    if HAS_STIM:
        def time_append_h_stim(self, num_qubits):
            for i in range(self.num_qubits):
                self.t_stim.append(STIM_H, [i])
        
        def time_append_s_stim(self, num_qubits):
            for i in range(self.num_qubits):
                self.t_stim.append(STIM_S, [i])
        
        def time_append_cnot_stim(self, num_qubits):
            for i in range(self.num_qubits - 1):
                self.t_stim.append(STIM_CNOT, [i, i + 1])
        
        def time_append_cz_stim(self, num_qubits):
            for i in range(self.num_qubits - 1):
                self.t_stim.append(STIM_CZ, [i, i + 1])


class CliffordUnitaryProperties:
    """Benchmark property checks."""
    params = [[10, 100]]
    param_names = ['num_qubits']
    
    def setup(self, num_qubits):
        self.c_identity_paulimer = CliffordUnitary.identity(num_qubits)
        self.c_nontrivial_paulimer = build_clifford_paulimer(num_qubits, variant=1)
        self.c_css_paulimer = CliffordUnitary.identity(num_qubits)
        for i in range(num_qubits):
            self.c_css_paulimer.left_mul(UnitaryOpcode.Hadamard, [i])
        if HAS_STIM:
            self.t_identity_stim = stim.Tableau(num_qubits)
            self.t_nontrivial_stim = build_clifford_stim(num_qubits, variant=1)
        if HAS_QISKIT:
            from qiskit import QuantumCircuit
            self.c_identity_qiskit = QiskitClifford(QuantumCircuit(num_qubits))
            self.c_nontrivial_qiskit = build_clifford_qiskit(num_qubits, variant=1)
    
    def time_is_identity_true_paulimer(self, num_qubits):
        _ = self.c_identity_paulimer.is_identity
    
    def time_is_identity_false_paulimer(self, num_qubits):
        _ = self.c_nontrivial_paulimer.is_identity
    
    def time_is_valid_paulimer(self, num_qubits):
        _ = self.c_nontrivial_paulimer.is_valid
    
    def time_is_css_paulimer(self, num_qubits):
        _ = self.c_css_paulimer.is_css
    
    def time_str_paulimer(self, num_qubits):
        _ = str(self.c_nontrivial_paulimer)
    
    def time_symplectic_matrix_paulimer(self, num_qubits):
        _ = self.c_nontrivial_paulimer.symplectic_matrix()
    
    if HAS_STIM:
        def time_str_stim(self, num_qubits):
            _ = str(self.t_nontrivial_stim)
        
        def time_to_numpy_stim(self, num_qubits):
            _ = self.t_nontrivial_stim.to_numpy()
    
    if HAS_QISKIT:
        def time_str_qiskit(self, num_qubits):
            _ = str(self.c_nontrivial_qiskit)
        
        def time_tableau_qiskit(self, num_qubits):
            _ = self.c_nontrivial_qiskit.tableau


class CliffordUnitaryTensor:
    """Benchmark tensor product operations."""
    params = [[5, 20], [5, 20]]
    param_names = ['num_qubits1', 'num_qubits2']
    
    def setup(self, num_qubits1, num_qubits2):
        self.c1_paulimer = build_clifford_paulimer(num_qubits1, variant=1)
        self.c2_paulimer = build_clifford_paulimer(num_qubits2, variant=2)
        if HAS_STIM:
            self.t1_stim = build_clifford_stim(num_qubits1, variant=1)
            self.t2_stim = build_clifford_stim(num_qubits2, variant=2)
        if HAS_QISKIT:
            self.c1_qiskit = build_clifford_qiskit(num_qubits1, variant=1)
            self.c2_qiskit = build_clifford_qiskit(num_qubits2, variant=2)
    
    def time_tensor_paulimer(self, num_qubits1, num_qubits2):
        _ = self.c1_paulimer.tensor(self.c2_paulimer)
    
    if HAS_STIM:
        def time_add_stim(self, num_qubits1, num_qubits2):
            _ = self.t1_stim + self.t2_stim
    
    if HAS_QISKIT:
        def time_tensor_qiskit(self, num_qubits1, num_qubits2):
            _ = self.c1_qiskit.tensor(self.c2_qiskit)
        
        def time_expand_qiskit(self, num_qubits1, num_qubits2):
            _ = self.c1_qiskit.expand(self.c2_qiskit)


class CliffordUnitarySerialization:
    """Benchmark serialization (pickle) operations."""
    params = [[10, 100]]
    param_names = ['num_qubits']
    
    def setup(self, num_qubits):
        self.c_paulimer = build_clifford_paulimer(num_qubits, variant=1)
        self.pickled_paulimer = pickle.dumps(self.c_paulimer)
        if HAS_STIM:
            self.t_stim = build_clifford_stim(num_qubits, variant=1)
            self.pickled_stim = pickle.dumps(self.t_stim)
        if HAS_QISKIT:
            self.c_qiskit = build_clifford_qiskit(num_qubits, variant=1)
            self.pickled_qiskit = pickle.dumps(self.c_qiskit)
    
    def time_dumps_paulimer(self, num_qubits):
        _ = pickle.dumps(self.c_paulimer)
    
    def time_loads_paulimer(self, num_qubits):
        _ = pickle.loads(self.pickled_paulimer)
    
    if HAS_STIM:
        def time_dumps_stim(self, num_qubits):
            _ = pickle.dumps(self.t_stim)
        
        def time_loads_stim(self, num_qubits):
            _ = pickle.loads(self.pickled_stim)
    
    if HAS_QISKIT:
        def time_dumps_qiskit(self, num_qubits):
            _ = pickle.dumps(self.c_qiskit)
        
        def time_loads_qiskit(self, num_qubits):
            _ = pickle.loads(self.pickled_qiskit)


class CliffordUnitaryMemoryFootprint:
    """Memory benchmarks to compare storage efficiency."""
    params = [[10, 100]]
    param_names = ['num_qubits']
    
    def setup(self, num_qubits):
        self.num_qubits = num_qubits
        if HAS_QISKIT:
            from qiskit import QuantumCircuit
            self.identity_circuit_qiskit = QuantumCircuit(num_qubits)
    
    def peakmem_create_identity_paulimer(self, num_qubits):
        _ = CliffordUnitary.identity(num_qubits)
    
    def peakmem_create_nontrivial_paulimer(self, num_qubits):
        _ = build_clifford_paulimer(self.num_qubits, variant=1)
    
    def peakmem_multiply_paulimer(self, num_qubits):
        c1 = build_clifford_paulimer(self.num_qubits, variant=1)
        c2 = build_clifford_paulimer(self.num_qubits, variant=2)
        _ = c1 * c2
    
    def peakmem_inverse_paulimer(self, num_qubits):
        c = build_clifford_paulimer(self.num_qubits, variant=1)
        _ = c.inverse()
    
    if HAS_STIM:
        def peakmem_create_identity_stim(self, num_qubits):
            _ = stim.Tableau(num_qubits)
        
        def peakmem_create_nontrivial_stim(self, num_qubits):
            _ = build_clifford_stim(self.num_qubits, variant=1)
        
        def peakmem_multiply_stim(self, num_qubits):
            t1 = build_clifford_stim(self.num_qubits, variant=1)
            t2 = build_clifford_stim(self.num_qubits, variant=2)
            _ = t1 * t2
        
        def peakmem_inverse_stim(self, num_qubits):
            t = build_clifford_stim(self.num_qubits, variant=1)
            _ = t.inverse()
    
    if HAS_QISKIT:
        def peakmem_create_identity_qiskit(self, num_qubits):
            _ = QiskitClifford(self.identity_circuit_qiskit)
        
        def peakmem_create_nontrivial_qiskit(self, num_qubits):
            _ = build_clifford_qiskit(self.num_qubits, variant=1)
        
        def peakmem_multiply_qiskit(self, num_qubits):
            c1 = build_clifford_qiskit(self.num_qubits, variant=1)
            c2 = build_clifford_qiskit(self.num_qubits, variant=2)
            _ = c1.compose(c2)
        
        def peakmem_inverse_qiskit(self, num_qubits):
            c = build_clifford_qiskit(self.num_qubits, variant=1)
            _ = c.adjoint()


class CliffordUnitaryCircuitSimulation:
    """Benchmark building up a Clifford unitary from a sequence of gates."""
    params = [[10, 50], [10, 100]]
    param_names = ['num_qubits', 'num_gates']
    
    def setup(self, num_qubits, num_gates):
        import random
        random.seed(42)
        
        self.gate_sequence = []
        for _ in range(num_gates):
            gate_type = random.choice(['H', 'S', 'CX', 'CZ'])
            if gate_type in ['H', 'S']:
                qubit = random.randint(0, num_qubits - 1)
                self.gate_sequence.append((gate_type, qubit))
            else:
                q1 = random.randint(0, num_qubits - 2)
                q2 = random.randint(q1 + 1, num_qubits - 1)
                self.gate_sequence.append((gate_type, q1, q2))
        
        if HAS_STIM:
            self.stim_circuit = stim.Circuit()
            for gate_info in self.gate_sequence:
                gate_type = gate_info[0]
                if gate_type == 'H':
                    self.stim_circuit.append('H', [gate_info[1]])
                elif gate_type == 'S':
                    self.stim_circuit.append('S', [gate_info[1]])
                elif gate_type == 'CX':
                    self.stim_circuit.append('CNOT', [gate_info[1], gate_info[2]])
                elif gate_type == 'CZ':
                    self.stim_circuit.append('CZ', [gate_info[1], gate_info[2]])
        
        if HAS_QISKIT:
            from qiskit import QuantumCircuit
            self.circuit_qiskit = QuantumCircuit(num_qubits)
            for gate_info in self.gate_sequence:
                gate_type = gate_info[0]
                if gate_type == 'H':
                    self.circuit_qiskit.h(gate_info[1])
                elif gate_type == 'S':
                    self.circuit_qiskit.s(gate_info[1])
                elif gate_type == 'CX':
                    self.circuit_qiskit.cx(gate_info[1], gate_info[2])
                elif gate_type == 'CZ':
                    self.circuit_qiskit.cz(gate_info[1], gate_info[2])
    
    def time_build_circuit_paulimer(self, num_qubits, num_gates):
        c = CliffordUnitary.identity(num_qubits)
        for gate_info in self.gate_sequence:
            gate_type = gate_info[0]
            if gate_type == 'H':
                c.left_mul(UnitaryOpcode.Hadamard, [gate_info[1]])
            elif gate_type == 'S':
                c.left_mul(UnitaryOpcode.SqrtZ, [gate_info[1]])
            elif gate_type == 'CX':
                c.left_mul(UnitaryOpcode.ControlledX, [gate_info[1], gate_info[2]])
            elif gate_type == 'CZ':
                c.left_mul(UnitaryOpcode.ControlledZ, [gate_info[1], gate_info[2]])
    
    if HAS_STIM:
        def time_build_circuit_stim(self, num_qubits, num_gates):
            t = stim.Tableau(num_qubits)
            for gate_info in self.gate_sequence:
                gate_type = gate_info[0]
                if gate_type == 'H':
                    t.append(STIM_H, [gate_info[1]])
                elif gate_type == 'S':
                    t.append(STIM_S, [gate_info[1]])
                elif gate_type == 'CX':
                    t.append(STIM_CNOT, [gate_info[1], gate_info[2]])
                elif gate_type == 'CZ':
                    t.append(STIM_CZ, [gate_info[1], gate_info[2]])
        
        def time_to_tableau_stim(self, num_qubits, num_gates):
            _ = self.stim_circuit.to_tableau()
    
    if HAS_QISKIT:
        def time_from_circuit_qiskit(self, num_qubits, num_gates):
            _ = QiskitClifford.from_circuit(self.circuit_qiskit)


class CliffordUnitaryBulkConjugation:
    """Benchmark conjugating many Pauli operators."""
    params = [[20, 100], [10, 100]]
    param_names = ['num_qubits', 'num_paulis']
    
    def setup(self, num_qubits, num_paulis):
        self.c_paulimer = build_clifford_paulimer(num_qubits, variant=1)
        self.paulis_paulimer = [SparsePauli(pauli_pattern(num_qubits, i)) for i in range(num_paulis)]
        if HAS_STIM:
            self.t_stim = build_clifford_stim(num_qubits, variant=1)
            self.paulis_stim = [stim.PauliString(pauli_pattern(num_qubits, i)) for i in range(num_paulis)]
        if HAS_QISKIT:
            self.c_qiskit = build_clifford_qiskit(num_qubits, variant=1)
            self.paulis_qiskit = [QiskitPauli(pauli_pattern(num_qubits, i)[::-1]) for i in range(num_paulis)]
    
    def time_conjugate_all_paulimer(self, num_qubits, num_paulis):
        for pauli in self.paulis_paulimer:
            _ = self.c_paulimer.image_of(pauli)
    
    if HAS_STIM:
        def time_conjugate_all_stim(self, num_qubits, num_paulis):
            for pauli in self.paulis_stim:
                _ = self.t_stim(pauli)
    
    if HAS_QISKIT:
        def time_conjugate_all_qiskit(self, num_qubits, num_paulis):
            for pauli in self.paulis_qiskit:
                _ = pauli.evolve(self.c_qiskit)


class CliffordUnitaryPower:
    """Benchmark computing powers of Clifford unitaries."""
    params = [[10, 100], [2, 16]]
    param_names = ['num_qubits', 'power']
    
    def setup(self, num_qubits, power):
        self.c_paulimer = build_clifford_paulimer(num_qubits, variant=1)
        if HAS_STIM:
            self.t_stim = build_clifford_stim(num_qubits, variant=1)
        if HAS_QISKIT:
            self.c_qiskit = build_clifford_qiskit(num_qubits, variant=1)
    
    def time_power_paulimer(self, num_qubits, power):
        _ = self.c_paulimer ** power
    
    if HAS_STIM:
        def time_power_stim(self, num_qubits, power):
            _ = self.t_stim ** power
    
    if HAS_QISKIT:
        def time_power_qiskit(self, num_qubits, power):
            _ = self.c_qiskit.power(power)


class CliffordUnitaryEquality:
    """Benchmark equality comparison of Clifford unitaries."""
    params = [[10, 100]]
    param_names = ['num_qubits']
    
    def setup(self, num_qubits):
        self.c1_paulimer = build_clifford_paulimer(num_qubits, variant=1)
        self.c2_paulimer = build_clifford_paulimer(num_qubits, variant=1)
        self.c3_paulimer = build_clifford_paulimer(num_qubits, variant=2)
        if HAS_STIM:
            self.t1_stim = build_clifford_stim(num_qubits, variant=1)
            self.t2_stim = build_clifford_stim(num_qubits, variant=1)
            self.t3_stim = build_clifford_stim(num_qubits, variant=2)
        if HAS_QISKIT:
            self.c1_qiskit = build_clifford_qiskit(num_qubits, variant=1)
            self.c2_qiskit = build_clifford_qiskit(num_qubits, variant=1)
            self.c3_qiskit = build_clifford_qiskit(num_qubits, variant=2)
    
    def time_eq_true_paulimer(self, num_qubits):
        _ = self.c1_paulimer == self.c2_paulimer
    
    def time_eq_false_paulimer(self, num_qubits):
        _ = self.c1_paulimer == self.c3_paulimer
    
    def time_ne_true_paulimer(self, num_qubits):
        _ = self.c1_paulimer != self.c3_paulimer
    
    def time_ne_false_paulimer(self, num_qubits):
        _ = self.c1_paulimer != self.c2_paulimer
    
    if HAS_STIM:
        def time_eq_true_stim(self, num_qubits):
            _ = self.t1_stim == self.t2_stim
        
        def time_eq_false_stim(self, num_qubits):
            _ = self.t1_stim == self.t3_stim
    
    if HAS_QISKIT:
        def time_eq_true_qiskit(self, num_qubits):
            _ = self.c1_qiskit == self.c2_qiskit
        
        def time_eq_false_qiskit(self, num_qubits):
            _ = self.c1_qiskit == self.c3_qiskit


class CliffordUnitarySpecialStructures:
    """Benchmark operations specific to paulimer (CSS, diagonal, etc.)."""
    params = [[10, 50]]
    param_names = ['num_qubits']
    
    def setup(self, num_qubits):
        self.c_css = CliffordUnitary.identity(num_qubits)
        for i in range(num_qubits):
            self.c_css.left_mul(UnitaryOpcode.Hadamard, [i])
        self.c_diagonal = CliffordUnitary.identity(num_qubits)
        for i in range(num_qubits - 1):
            self.c_diagonal.left_mul(UnitaryOpcode.ControlledZ, [i, i + 1])
    
    def time_is_css_paulimer(self, num_qubits):
        _ = self.c_css.is_css
    
    def time_is_diagonal_x_paulimer(self, num_qubits):
        _ = self.c_diagonal.is_diagonal("X")
    
    def time_is_diagonal_z_paulimer(self, num_qubits):
        _ = self.c_diagonal.is_diagonal("Z")
    
    def time_is_diagonal_resource_encoder_z_paulimer(self, num_qubits):
        _ = is_diagonal_resource_encoder(self.c_diagonal, "Z")
    
    def time_split_qubit_cliffords_and_css_paulimer(self, num_qubits):
        _ = split_qubit_cliffords_and_css(self.c_css)
    
    def time_split_phased_css_paulimer(self, num_qubits):
        _ = split_phased_css(self.c_css)
