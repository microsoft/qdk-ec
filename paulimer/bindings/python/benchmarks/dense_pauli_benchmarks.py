"""
Benchmarks for DensePauli operations, comparing against Stim's PauliString, fast_pauli's PauliString, and Qiskit's Pauli.
"""

import pickle

try:
    import stim
    HAS_STIM = True
except ImportError:
    HAS_STIM = False

try:
    from fast_pauli import PauliString as FastPauliString
    HAS_FAST_PAULI = True
except ImportError:
    HAS_FAST_PAULI = False

try:
    from qiskit.quantum_info import Pauli as QiskitPauli
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

from paulimer import DensePauli


def pauli_pattern(num_qubits, offset=0):
    """Generate a deterministic Pauli pattern string like 'IXYZIXYZ...'."""
    return "".join(["IXYZ"[(offset + j) % 4] for j in range(num_qubits)])


class DensePauliInitialization:
    """Benchmark initialization of Pauli operators."""
    params = [[10, 1000]]
    param_names = ['num_qubits']
    
    def setup(self, num_qubits):
        self.pauli_string_x = "X" * num_qubits
        self.pauli_string_mixed = pauli_pattern(num_qubits)
        self.pauli_string_sparse = "I" * (num_qubits - 5) + "XYZXY"
        self.pauli_string_identity = "I" * num_qubits
        self.pauli_string_x_rev = self.pauli_string_x[::-1]
        self.pauli_string_mixed_rev = self.pauli_string_mixed[::-1]
        self.pauli_string_sparse_rev = self.pauli_string_sparse[::-1]
    
    def time_from_string_all_x_paulimer(self, num_qubits):
        DensePauli(self.pauli_string_x)
    
    def time_from_string_mixed_paulimer(self, num_qubits):
        DensePauli(self.pauli_string_mixed)
    
    def time_from_string_sparse_paulimer(self, num_qubits):
        DensePauli(self.pauli_string_sparse)
    
    def time_identity_paulimer(self, num_qubits):
        DensePauli.identity(num_qubits)
    
    def time_single_x_paulimer(self, num_qubits):
        DensePauli.x(num_qubits // 2, num_qubits)
    
    def time_single_y_paulimer(self, num_qubits):
        DensePauli.y(num_qubits // 2, num_qubits)
    
    def time_single_z_paulimer(self, num_qubits):
        DensePauli.z(num_qubits // 2, num_qubits)
    
    if HAS_STIM:
        def time_from_string_all_x_stim(self, num_qubits):
            stim.PauliString(self.pauli_string_x)
        
        def time_from_string_mixed_stim(self, num_qubits):
            stim.PauliString(self.pauli_string_mixed)
        
        def time_from_string_sparse_stim(self, num_qubits):
            stim.PauliString(self.pauli_string_sparse)
        
        def time_identity_stim(self, num_qubits):
            stim.PauliString(num_qubits)
    
    if HAS_FAST_PAULI:
        def time_from_string_all_x_fast_pauli(self, num_qubits):
            FastPauliString(self.pauli_string_x)
        
        def time_from_string_mixed_fast_pauli(self, num_qubits):
            FastPauliString(self.pauli_string_mixed)
        
        def time_from_string_sparse_fast_pauli(self, num_qubits):
            FastPauliString(self.pauli_string_sparse)
        
        def time_identity_fast_pauli(self, num_qubits):
            FastPauliString(self.pauli_string_identity)
    
    if HAS_QISKIT:
        def time_from_string_all_x_qiskit(self, num_qubits):
            QiskitPauli(self.pauli_string_x_rev)
        
        def time_from_string_mixed_qiskit(self, num_qubits):
            QiskitPauli(self.pauli_string_mixed_rev)
        
        def time_from_string_sparse_qiskit(self, num_qubits):
            QiskitPauli(self.pauli_string_sparse_rev)
        
        def time_identity_qiskit(self, num_qubits):
            QiskitPauli(self.pauli_string_identity)


class DensePauliMultiplication:
    """Benchmark multiplication of Pauli operators."""
    params = [[10, 1000]]
    param_names = ['num_qubits']
    
    def setup(self, num_qubits):
        self.p1_paulimer = DensePauli(pauli_pattern(num_qubits, 0))
        self.p2_paulimer = DensePauli(pauli_pattern(num_qubits, 1))
        if HAS_STIM:
            self.p1_stim = stim.PauliString(pauli_pattern(num_qubits, 0))
            self.p2_stim = stim.PauliString(pauli_pattern(num_qubits, 1))
        if HAS_FAST_PAULI:
            self.p1_fast_pauli = FastPauliString(pauli_pattern(num_qubits, 0))
            self.p2_fast_pauli = FastPauliString(pauli_pattern(num_qubits, 1))
        if HAS_QISKIT:
            self.p1_qiskit = QiskitPauli(pauli_pattern(num_qubits, 0)[::-1])
            self.p2_qiskit = QiskitPauli(pauli_pattern(num_qubits, 1)[::-1])
    
    def time_mul_paulimer(self, num_qubits):
        _ = self.p1_paulimer * self.p2_paulimer
    
    def time_imul_paulimer(self, num_qubits):
        self.p1_paulimer *= self.p2_paulimer
    
    if HAS_STIM:
        def time_mul_stim(self, num_qubits):
            _ = self.p1_stim * self.p2_stim
        
        def time_imul_stim(self, num_qubits):
            self.p1_stim *= self.p2_stim
    
    if HAS_FAST_PAULI:
        def time_matmul_fast_pauli(self, num_qubits):
            _ = self.p1_fast_pauli @ self.p2_fast_pauli
    
    if HAS_QISKIT:
        def time_compose_qiskit(self, num_qubits):
            _ = self.p1_qiskit.compose(self.p2_qiskit)


class DensePauliCommutation:
    """Benchmark commutation checks."""
    params = [[10, 1000]]
    param_names = ['num_qubits']
    
    def setup(self, num_qubits):
        self.p1_paulimer = DensePauli("X" * num_qubits)
        self.p2_paulimer_commute = DensePauli("X" * num_qubits)
        self.p2_paulimer_anticommute = DensePauli("Z" * num_qubits)
        if HAS_STIM:
            self.p1_stim = stim.PauliString("X" * num_qubits)
            self.p2_stim_commute = stim.PauliString("X" * num_qubits)
            self.p2_stim_anticommute = stim.PauliString("Z" * num_qubits)
        if HAS_QISKIT:
            self.p1_qiskit = QiskitPauli("X" * num_qubits)
            self.p2_qiskit_commute = QiskitPauli("X" * num_qubits)
            self.p2_qiskit_anticommute = QiskitPauli("Z" * num_qubits)
    
    def time_commutes_true_paulimer(self, num_qubits):
        self.p1_paulimer.commutes_with(self.p2_paulimer_commute)
    
    def time_commutes_false_paulimer(self, num_qubits):
        self.p1_paulimer.commutes_with(self.p2_paulimer_anticommute)
    
    if HAS_STIM:
        def time_commutes_true_stim(self, num_qubits):
            self.p1_stim.commutes(self.p2_stim_commute)
        
        def time_commutes_false_stim(self, num_qubits):
            self.p1_stim.commutes(self.p2_stim_anticommute)
    
    if HAS_QISKIT:
        def time_commutes_true_qiskit(self, num_qubits):
            self.p1_qiskit.commutes(self.p2_qiskit_commute)
        
        def time_commutes_false_qiskit(self, num_qubits):
            self.p1_qiskit.commutes(self.p2_qiskit_anticommute)


class DensePauliProperties:
    """Benchmark property computations."""
    params = [[10, 1000]]
    param_names = ['num_qubits']
    
    def setup(self, num_qubits):
        pattern = pauli_pattern(num_qubits)
        self.p_paulimer = DensePauli(pattern)
        if HAS_STIM:
            self.p_stim = stim.PauliString(pattern)
        if HAS_FAST_PAULI:
            self.p_fast_pauli = FastPauliString(pattern)
        if HAS_QISKIT:
            self.p_qiskit = QiskitPauli(pattern[::-1])
    
    def time_weight_paulimer(self, num_qubits):
        _ = self.p_paulimer.weight
    
    def time_support_paulimer(self, num_qubits):
        _ = self.p_paulimer.support
    
    def time_str_paulimer(self, num_qubits):
        _ = str(self.p_paulimer)
    
    if HAS_STIM:
        def time_weight_stim(self, num_qubits):
            _ = self.p_stim.weight
        
        def time_str_stim(self, num_qubits):
            _ = str(self.p_stim)
    
    if HAS_FAST_PAULI:
        def time_weight_fast_pauli(self, num_qubits):
            _ = self.p_fast_pauli.weight
        
        def time_str_fast_pauli(self, num_qubits):
            _ = str(self.p_fast_pauli)
    
    if HAS_QISKIT:
        def time_str_qiskit(self, num_qubits):
            _ = str(self.p_qiskit)


class DensePauliIndexing:
    """Benchmark indexing operations."""
    params = [[10, 1000]]
    param_names = ['num_qubits']
    
    def setup(self, num_qubits):
        pattern = pauli_pattern(num_qubits)
        self.num_qubits = num_qubits
        self.p_paulimer = DensePauli(pattern)
        if HAS_STIM:
            self.p_stim = stim.PauliString(pattern)
        if HAS_FAST_PAULI:
            self.p_fast_pauli = FastPauliString(pattern)
        if HAS_QISKIT:
            self.p_qiskit = QiskitPauli(pattern[::-1])
    
    def time_getitem_start_paulimer(self, num_qubits):
        _ = self.p_paulimer[0]
    
    def time_getitem_middle_paulimer(self, num_qubits):
        _ = self.p_paulimer[self.num_qubits // 2]
    
    def time_getitem_end_paulimer(self, num_qubits):
        _ = self.p_paulimer[self.num_qubits - 1]
    
    if HAS_STIM:
        def time_getitem_start_stim(self, num_qubits):
            _ = self.p_stim[0]
        
        def time_getitem_middle_stim(self, num_qubits):
            _ = self.p_stim[self.num_qubits // 2]
        
        def time_getitem_end_stim(self, num_qubits):
            _ = self.p_stim[self.num_qubits - 1]
    
    if HAS_QISKIT:
        def time_getitem_start_qiskit(self, num_qubits):
            _ = self.p_qiskit[0]
        
        def time_getitem_middle_qiskit(self, num_qubits):
            _ = self.p_qiskit[self.num_qubits // 2]
        
        def time_getitem_end_qiskit(self, num_qubits):
            _ = self.p_qiskit[self.num_qubits - 1]


class DensePauliComparison:
    """Benchmark equality comparisons."""
    params = [[10, 1000]]
    param_names = ['num_qubits']
    
    def setup(self, num_qubits):
        pattern = pauli_pattern(num_qubits, 0)
        pattern_diff = pauli_pattern(num_qubits, 1)
        self.p1_paulimer = DensePauli(pattern)
        self.p2_paulimer_same = DensePauli(pattern)
        self.p2_paulimer_diff = DensePauli(pattern_diff)
        if HAS_STIM:
            self.p1_stim = stim.PauliString(pattern)
            self.p2_stim_same = stim.PauliString(pattern)
            self.p2_stim_diff = stim.PauliString(pattern_diff)
        if HAS_FAST_PAULI:
            self.p1_fast_pauli = FastPauliString(pattern)
            self.p2_fast_pauli_same = FastPauliString(pattern)
            self.p2_fast_pauli_diff = FastPauliString(pattern_diff)
        if HAS_QISKIT:
            self.p1_qiskit = QiskitPauli(pattern[::-1])
            self.p2_qiskit_same = QiskitPauli(pattern[::-1])
            self.p2_qiskit_diff = QiskitPauli(pattern_diff[::-1])
    
    def time_eq_true_paulimer(self, num_qubits):
        _ = self.p1_paulimer == self.p2_paulimer_same
    
    def time_eq_false_paulimer(self, num_qubits):
        _ = self.p1_paulimer == self.p2_paulimer_diff
    
    if HAS_STIM:
        def time_eq_true_stim(self, num_qubits):
            _ = self.p1_stim == self.p2_stim_same
        
        def time_eq_false_stim(self, num_qubits):
            _ = self.p1_stim == self.p2_stim_diff
    
    if HAS_QISKIT:
        def time_eq_true_qiskit(self, num_qubits):
            _ = self.p1_qiskit == self.p2_qiskit_same
        
        def time_eq_false_qiskit(self, num_qubits):
            _ = self.p1_qiskit == self.p2_qiskit_diff


class DensePauliSerialization:
    """Benchmark serialization (pickle) operations."""
    params = [[10, 1000]]
    param_names = ['num_qubits']
    
    def setup(self, num_qubits):
        pattern = pauli_pattern(num_qubits)
        self.p_paulimer = DensePauli(pattern)
        self.pickled_paulimer = pickle.dumps(self.p_paulimer)
        if HAS_STIM:
            self.p_stim = stim.PauliString(pattern)
            self.pickled_stim = pickle.dumps(self.p_stim)
        if HAS_FAST_PAULI:
            self.p_fast_pauli = FastPauliString(pattern)
            self.pickled_fast_pauli = pickle.dumps(self.p_fast_pauli)
        if HAS_QISKIT:
            self.p_qiskit = QiskitPauli(pattern[::-1])
            self.pickled_qiskit = pickle.dumps(self.p_qiskit)
    
    def time_dumps_paulimer(self, num_qubits):
        _ = pickle.dumps(self.p_paulimer)
    
    def time_loads_paulimer(self, num_qubits):
        _ = pickle.loads(self.pickled_paulimer)
    
    if HAS_STIM:
        def time_dumps_stim(self, num_qubits):
            _ = pickle.dumps(self.p_stim)
        
        def time_loads_stim(self, num_qubits):
            _ = pickle.loads(self.pickled_stim)
    
    if HAS_FAST_PAULI:
        def time_dumps_fast_pauli(self, num_qubits):
            _ = pickle.dumps(self.p_fast_pauli)
        
        def time_loads_fast_pauli(self, num_qubits):
            _ = pickle.loads(self.pickled_fast_pauli)
    
    if HAS_QISKIT:
        def time_dumps_qiskit(self, num_qubits):
            _ = pickle.dumps(self.p_qiskit)
        
        def time_loads_qiskit(self, num_qubits):
            _ = pickle.loads(self.pickled_qiskit)


class DensePauliMemoryFootprint:
    """Memory benchmarks to compare storage efficiency."""
    params = [[100, 5000]]
    param_names = ['num_qubits']
    
    def setup(self, num_qubits):
        self.pattern = pauli_pattern(num_qubits)
    
    def peakmem_create_paulimer(self, num_qubits):
        _ = DensePauli(self.pattern)
    
    def peakmem_multiply_paulimer(self, num_qubits):
        p1 = DensePauli(self.pattern)
        p2 = DensePauli(self.pattern)
        _ = p1 * p2
    
    if HAS_STIM:
        def peakmem_create_stim(self, num_qubits):
            _ = stim.PauliString(self.pattern)
        
        def peakmem_multiply_stim(self, num_qubits):
            p1 = stim.PauliString(self.pattern)
            p2 = stim.PauliString(self.pattern)
            _ = p1 * p2
    
    if HAS_FAST_PAULI:
        def peakmem_create_fast_pauli(self, num_qubits):
            _ = FastPauliString(self.pattern)
    
    if HAS_QISKIT:
        def peakmem_create_qiskit(self, num_qubits):
            _ = QiskitPauli(self.pattern[::-1])


class DensePauliBulkOperations:
    """Benchmark bulk operations on many Pauli operators."""
    params = [[100, 1000], [10, 100]]
    param_names = ['num_qubits', 'num_paulis']
    
    def setup(self, num_qubits, num_paulis):
        self.paulis_paulimer = [DensePauli(pauli_pattern(num_qubits, j)) for j in range(num_paulis)]
        self.result_paulimer = DensePauli(pauli_pattern(num_qubits, 0))
        if HAS_STIM:
            self.paulis_stim = [stim.PauliString(pauli_pattern(num_qubits, j)) for j in range(num_paulis)]
            self.result_stim = stim.PauliString(pauli_pattern(num_qubits, 0))
        if HAS_FAST_PAULI:
            self.paulis_fast_pauli = [FastPauliString(pauli_pattern(num_qubits, j)) for j in range(num_paulis)]
        if HAS_QISKIT:
            self.paulis_qiskit = [QiskitPauli(pauli_pattern(num_qubits, j)[::-1]) for j in range(num_paulis)]
            self.result_qiskit = QiskitPauli(pauli_pattern(num_qubits, 0)[::-1])
    
    def time_sequential_multiply_paulimer(self, num_qubits, num_paulis):
        for p in self.paulis_paulimer:
            self.result_paulimer *= p
    
    def time_commutation_matrix_paulimer(self, num_qubits, num_paulis):
        for i, p1 in enumerate(self.paulis_paulimer):
            for p2 in self.paulis_paulimer[i+1:]:
                _ = p1.commutes_with(p2)
    
    if HAS_STIM:
        def time_sequential_multiply_stim(self, num_qubits, num_paulis):
            for p in self.paulis_stim:
                self.result_stim *= p
        
        def time_commutation_matrix_stim(self, num_qubits, num_paulis):
            for i, p1 in enumerate(self.paulis_stim):
                for p2 in self.paulis_stim[i+1:]:
                    _ = p1.commutes(p2)
    
    if HAS_FAST_PAULI:
        def time_sequential_matmul_fast_pauli(self, num_qubits, num_paulis):
            result = self.paulis_fast_pauli[0]
            for p in self.paulis_fast_pauli[1:]:
                _, result = result @ p
    
    if HAS_QISKIT:
        def time_sequential_compose_qiskit(self, num_qubits, num_paulis):
            for p in self.paulis_qiskit:
                self.result_qiskit = self.result_qiskit.compose(p)
        
        def time_commutation_matrix_qiskit(self, num_qubits, num_paulis):
            for i, p1 in enumerate(self.paulis_qiskit):
                for p2 in self.paulis_qiskit[i+1:]:
                    _ = p1.commutes(p2)
