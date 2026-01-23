from enum import IntEnum
from typing import (
    Literal,
    final,
    Optional,
    Any,
    Iterable,
    Protocol,
    Sequence,
    overload,
)
from binar import BitMatrix, BitVector

PauliCharacter = Literal["I", "X", "Y", "Z"]
Exponent = int

class UnitaryOpcode(IntEnum):
    """Enum of standard Clifford gates and operations.
    
    Opcodes represent common single and two-qubit Clifford gates used in
    quantum circuits. Use with CliffordUnitary.from_name() or simulation
    methods like apply_unitary().
    
    Examples:
        >>> UnitaryOpcode.Hadamard
        >>> UnitaryOpcode.from_string("CNOT")
        >>> UnitaryOpcode.ControlledX  # Equivalent to CNOT
    """
    I = 0
    X = 1
    Y = 2
    Z = 3
    SqrtX = 4
    SqrtXInv = 5
    SqrtY = 6
    SqrtYInv = 7
    SqrtZ = 8
    SqrtZInv = 9
    Hadamard = 10
    Swap = 11
    ControlledX = 12
    ControlledZ = 13
    PrepareBell = 14

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    @staticmethod
    def from_string(s: str) -> "UnitaryOpcode": ...

@final
class DensePauli:
    """Dense representation of a Pauli operator on a fixed number of qubits.
    
    Stores a Pauli operator as a dense string of characters (e.g., "IXYZ") with
    an associated phase. Efficient for dense operators or when qubit count is fixed.
    
    Phase convention: Pauli operators are represented as exp(iπ*exponent/4) * P
    where P is a tensor product of X, Y, Z operators.
    
    Examples:
        >>> p = DensePauli("XYZ")
        >>> p.weight
        3
        >>> p * DensePauli("YXI")
        DensePauli("-iZZI")
    """
    def __init__(self, characters: str="") -> None:
        """Create a DensePauli from a character string.
        
        Args:
            characters: String of Pauli characters (I, X, Y, Z), optionally
                       prefixed with phase (1, i, -1, -i).
        
        Examples:
            >>> DensePauli("XYZ")
            >>> DensePauli("iXYZ")  # Phase i
            >>> DensePauli("-XYZ")  # Phase -1
        """
        ...
    @staticmethod
    def identity(size: int) -> "DensePauli": ...
    @staticmethod
    def x(index: int, size: int) -> "DensePauli": ...
    @staticmethod
    def y(index: int, size: int) -> "DensePauli": ...
    @staticmethod
    def z(index: int, size: int) -> "DensePauli": ...
    @staticmethod
    def from_sparse(pauli: "SparsePauli", size: int) -> "DensePauli": ...
    @property
    def exponent(self) -> Exponent: 
        ### The value of `exponent`, when `self` is written in the form e**(iπ * exponent / 4) XᵃZᵇ.
        ...
    @property
    def phase(self) -> complex: 
        ### The complex phase of `self` when written in tensor product form e**(iπθ) P₁⊗P₂..., i.e., one of {1, i, -1, -i}.
        ...
    @property
    def characters(self) -> str:
        """String representation without phase (e.g., \"IXYZ\")."""
        ...
    @property
    def support(self) -> list[int]:
        """Indices of non-identity Pauli operators."""
        ...
    @property
    def weight(self) -> int:
        """Number of non-identity Pauli operators."""
        ...
    @property
    def size(self) -> int:
        """Total number of qubits."""
        ...
    def commutes_with(self, other: "DensePauli" | Iterable["DensePauli"]) -> bool:
        """Check if this operator commutes with another or collection of operators.
        
        Args:
            other: Single DensePauli or iterable of DensePauli operators.
        
        Returns:
            True if all operators commute with self.
        """
        ...
    def copy(self) -> "DensePauli": ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __mul__(self, other: "DensePauli") -> "DensePauli": ...
    def __imul__(self, other: "DensePauli") -> "DensePauli": ...
    def __add__(self, other: "DensePauli") -> "DensePauli": ...
    def __abs__(self) -> "DensePauli": ...
    def __neg__(self) -> "DensePauli": ...
    def __getitem__(self, index: int) -> PauliCharacter: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __getstate__(self) -> tuple: ...
    def __setstate__(self, state: tuple) -> None: ...

@final
class SparsePauli:
    """Sparse representation of a Pauli operator.
    
    Stores only the non-identity Pauli operators with their qubit indices.
    Efficient for operators with small weight, especially in large systems.
    
    Phase convention: Same as DensePauli - exp(iπ*exponent/4) * P.
    
    Examples:
        >>> p = SparsePauli("X_2 Z_5")  # X on qubit 2, Z on qubit 5
        >>> p.support
        (2, 5)
        >>> SparsePauli({2: "X", 5: "Z"})
    """
    @overload
    def __init__(self, characters: str = "") -> None:
        """Create from string like \"X_2 Z_5\" or \"X2Z5\"."""
        ...
    @overload
    def __init__(self, characters: dict[int, PauliCharacter], exponent: Exponent = 0) -> None:
        """Create from dict mapping qubit indices to Pauli characters.
        
        Args:
            characters: Dict like {0: \"X\", 3: \"Z\"}.
            exponent: Phase exponent (0, 1, 2, 3 for phases 1, i, -1, -i).
        """
        ...
    @staticmethod
    def identity() -> "SparsePauli": ...
    @staticmethod
    def x(index: int) -> "SparsePauli": ...
    @staticmethod
    def y(index: int) -> "SparsePauli": ...
    @staticmethod
    def z(index: int) -> "SparsePauli": ...
    @staticmethod
    def from_dense(dense_pauli: DensePauli) -> "SparsePauli": ...
    @property
    def exponent(self) -> Exponent:
        """Phase exponent where phase = exp(iπ*exponent/4)."""
        ...
    @property
    def phase(self) -> complex:
        """Complex phase (one of 1, i, -1, -i)."""
        ...
    @property
    def support(self) -> tuple[int]:
        """Tuple of qubit indices with non-identity operators."""
        ...
    @property
    def characters(self) -> str:
        """String representation without phase."""
        ...
    @property
    def weight(self) -> int:
        """Number of non-identity operators."""
        ...
    def commutes_with(self, other: "SparsePauli" | Iterable["SparsePauli"]) -> bool:
        """Check if this operator commutes with another or collection of operators."""
        ...
    def copy(self) -> "SparsePauli": ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __mul__(self, other: "SparsePauli") -> "SparsePauli": ...
    def __imul__(self, other: "SparsePauli") -> "SparsePauli": ...
    def __abs__(self) -> "SparsePauli": ...
    def __neg__(self) -> "SparsePauli": ...
    def __getitem__(self, index: int) -> PauliCharacter: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __getstate__(self) -> tuple: ...
    def __setstate__(self, state: tuple) -> None: ...


@final
class PauliGroup:
    """Group of Pauli operators generated by a set of generators.
    
    Represents a subgroup of the Pauli group, useful for stabilizer codes,
    normalizer computations, and group-theoretic analysis.
    
    Examples:
        >>> g1 = SparsePauli("X_0 X_1")
        >>> g2 = SparsePauli("Z_0 Z_1")
        >>> group = PauliGroup([g1, g2])
        >>> SparsePauli("Y_0 Y_1") in group
        True
    """
    def __init__(self, generators: Iterable[SparsePauli], all_commute: Optional[bool] = None) -> None:
        """Create a Pauli group from generators.
        
        Args:
            generators: Iterable of SparsePauli generators.
            all_commute: Hint whether all generators commute (optional optimization).
        """
        ...
    def factorization_of(self, element: SparsePauli) -> Optional[list[SparsePauli]]: ...
    def factorizations_of(self, elements: Iterable[SparsePauli]) -> list[Optional[list[SparsePauli]]]: ...
    def __contains__(self, element: SparsePauli) -> bool: ...
    def __eq__(self, other: Any) -> bool: ...
    def __le__(self, other: "PauliGroup") -> bool: ...
    def __lt__(self, other: "PauliGroup") -> bool: ...
    def __or__(self, other: "PauliGroup") -> "PauliGroup": ...
    def __and__(self, other: "PauliGroup") -> "PauliGroup": ...
    def __truediv__(self, other: "PauliGroup") -> "PauliGroup": ...
    def __getstate__(self) -> tuple: ...
    def __setstate__(self, state: tuple) -> None: ...
    
    @property
    def generators(self) -> list[SparsePauli]:
        """The set of generators."""
        ...
    @property
    def standard_generators(self) -> list[SparsePauli]:
        """Standard form generators (reduced form)."""
        ...
    @property
    def elements(self) -> Iterable[SparsePauli]:
        """Iterator over all group elements (may be large!)."""
        ...
    @property
    def phases(self) -> list[Exponent]:
        """Pure phases contained in the group."""
        ...
    @property
    def binary_rank(self) -> int:
        """Rank of the group's binary representation."""
        ...
    @property
    def support(self) -> list[int]:
        """Qubit indices touched by any generator."""
        ...
    @property
    def log2_size(self) -> int:
        """Log base 2 of the group size (number of independent generators)."""
        ...
    @property
    def is_abelian(self) -> bool:
        """True if all generators commute."""
        ...
    @property
    def is_stabilizer_group(self) -> bool:
        """True if this is a valid stabilizer group (abelian, no -I)."""
        ...


def centralizer_of(group: PauliGroup, supported_by: Optional[Iterable[int]]=None) -> PauliGroup:
    """Compute the centralizer of a Pauli group.
    
    Args:
        group: PauliGroup to centralize.
        supported_by: Optional restriction to specific qubits.
    
    Returns:
        PauliGroup of operators that commute with all elements of group.
    """
    ...
def symplectic_form_of(generators: Iterable[SparsePauli]) -> Iterable[SparsePauli]:
    """Compute symplectic form of a set of Pauli operators.
    
    Returns:
        Canonicalized generators in symplectic form.
    """
    ...

@final
class CliffordUnitary:
    """Clifford unitary operator on qubits.
    
    Represents a unitary in the Clifford group, stored as mappings of
    Pauli operators (by conjugation). Efficient for stabilizer simulation
    and circuit synthesis.
    
    Examples:
        >>> h = CliffordUnitary.from_name("Hadamard", [0], 1)
        >>> h.image_x(0)
        DensePauli("Z")
        >>> cnot = CliffordUnitary.from_name("CNOT", [0, 1], 2)
    """
    @staticmethod
    def from_string(characters: str) -> "CliffordUnitary":
        """Create from string representation.
        
        For example, creates one qubit Hadamard from string \"X_0:Z_0, Z_0:X_0\".
        """
        ...
    @staticmethod
    def from_preimages(preimages: Sequence[DensePauli]) -> "CliffordUnitary":
        """Create from preimages of the X and Z generators.
        
        Args:
            preimages: Sequence of Pauli operators [X_0', ..., X_{n-1}', Z_0', ..., Z_{n-1}'].
        """
        ...
    @staticmethod
    def from_images(images: Sequence[DensePauli]) -> "CliffordUnitary":
        """Create from images of the X and Z generators.
        
        Args:
            images: Sequence of Pauli operators [X_0, ..., X_{n-1}, Z_0, ..., Z_{n-1}].
        """
        ...
    @staticmethod
    def from_symplectic_matrix(matrix: BitMatrix) -> "CliffordUnitary":
        """Create from symplectic matrix representation."""
        ...
    @staticmethod
    def from_name(name: str, operands: Sequence[int], qubit_count: int) -> "CliffordUnitary":
        """Create a named gate.
        
        Args:
            name: Gate name (e.g., \"Hadamard\", \"CNOT\", \"S\").
            operands: Qubit indices.
            qubit_count: Total number of qubits.
        """
        ...
    @staticmethod
    def identity(qubit_count: int) -> "CliffordUnitary":
        """Create the identity on `qubit_count` qubits."""
        ...

    @property
    def is_css(self) -> bool:
        """True if this is a CSS (Calderbank-Shor-Steane) code unitary."""
        ...
    @property
    def qubit_count(self) -> int:
        """Number of qubits."""
        ...
    @property
    def is_valid(self) -> bool:
        """True if the internal representation is valid."""
        ...
    @property
    def is_identity(self) -> bool:
        """True if this is the identity operator."""
        ...

    def preimage_of(self, pauli: DensePauli | SparsePauli) -> DensePauli:
        """Compute U^† P U for a Pauli operator P."""
        ...
    def preimage_x(self, index: int) -> DensePauli:
        """Preimage of X_i."""
        ...
    def preimage_z(self, index: int) -> DensePauli:
        """Preimage of Z_i."""
        ...
    def image_of(self, pauli: DensePauli | SparsePauli) -> DensePauli:
        """Compute U P U^† for a Pauli operator P."""
        ...
    def image_x(self, index: int) -> DensePauli:
        """Image of X_i."""
        ...
    def image_z(self, index: int) -> DensePauli:
        """Image of Z_i."""
        ...
    def tensor(self, rhs: "CliffordUnitary") -> "CliffordUnitary":
        """Tensor product with another Clifford unitary."""
        ...
    def inverse(self) -> "CliffordUnitary":
        """Compute the inverse."""
        ...
    def is_diagonal(self, axis: Literal["X", "Z"]) -> bool:
        """Check if diagonal in the given axis basis."""
        ...
    def symplectic_matrix(self) -> BitMatrix:
        """Get the symplectic matrix representation."""
        ...
    def __mul__(self, other: "CliffordUnitary") -> "CliffordUnitary": ...
    def left_mul(self, opcode: UnitaryOpcode, operands: Sequence[int]) -> None: ...
    def left_mul_clifford(
        self, clifford: "CliffordUnitary", support: Sequence[int]
    ) -> None: ...
    def left_mul_permutation(
        self, permutation: Sequence[int], support: Sequence[int]
    ) -> None: ...
    def left_mul_pauli(self, pauli: DensePauli | SparsePauli) -> None: ...
    def left_mul_pauli_exp(self, pauli: DensePauli | SparsePauli) -> None: ...
    def left_mul_controlled_pauli(
        self, control: DensePauli | SparsePauli, target: DensePauli | SparsePauli
    ) -> None: ...
    def __pow__(self, exponent: int) -> "CliffordUnitary": ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __getstate__(self) -> tuple: ...
    def __setstate__(self, state: tuple) -> None: ...


def is_diagonal_resource_encoder(clifford: CliffordUnitary, axis: Literal["X", "Z"]) -> bool:
    """Check if a Clifford encodes a diagonal resource state."""
    ...
def unitary_from_diagonal_resource_state(clifford: CliffordUnitary, axis: Literal["X", "Z"]) -> "CliffordUnitary" | None:
    """Extract unitary from a diagonal resource state encoder."""
    ...
def split_qubit_cliffords_and_css(clifford: CliffordUnitary) -> tuple["CliffordUnitary", "CliffordUnitary"] | None:
    """Split into single-qubit Cliffords and CSS components."""
    ...
def split_phased_css(clifford: CliffordUnitary) -> tuple["CliffordUnitary", "CliffordUnitary"] | None:
    """Split into phased CSS components."""
    ...
def encoding_clifford_of(generators: Sequence[SparsePauli | DensePauli], qubit_count: int) -> "CliffordUnitary":
    """Construct encoding Clifford from stabilizer generators.
    
    Args:
        generators: Stabilizer generators.
        qubit_count: Total number of qubits.
    
    Returns:
        Clifford unitary that maps logical Paulis to given generators.
    """
    ...


class StabilizerSimulation(Protocol):
    """Protocol for stabilizer simulation interfaces.
    
    Defines the common interface implemented by OutcomeCompleteSimulation,
    OutcomeFreeSimulation, OutcomeSpecificSimulation, and FaultySimulation.
    
    Use this for type hints when writing code that works with any simulation type.
    """

    @property
    def qubit_count(self) -> int:
        """Current number of qubits in use."""
        ...
    @property
    def qubit_capacity(self) -> int:
        """Maximum qubits before reallocation."""
        ...
    @property
    def outcome_count(self) -> int:
        """Number of measurement outcomes recorded."""
        ...
    @property
    def outcome_capacity(self) -> int:
        """Maximum outcomes before reallocation."""
        ...
    @property
    def random_outcome_count(self) -> int:
        """Number of random outcome bits."""
        ...
    @property
    def random_outcome_capacity(self) -> int:
        """Maximum random outcomes."""
        ...
    @property
    def random_bit_count(self) -> int:
        """Number of random bits consumed."""
        ...

    def apply_unitary(self, unitary_op: UnitaryOpcode, support: Sequence[int]) -> None: ...
    def apply_pauli_exp(self, observable: SparsePauli) -> None: ...
    def apply_pauli(self, observable: SparsePauli, controlled_by: SparsePauli | None = None) -> None: ...
    def apply_conditional_pauli(
        self, observable: SparsePauli, outcomes: Sequence[int], parity: bool = True,
    ) -> None: ...
    def apply_permutation(self, permutation: Sequence[int], supported_by: Sequence[int] | None = None) -> None: ...
    def apply_clifford(self, clifford: CliffordUnitary, supported_by: Sequence[int] | None = None) -> None: ...
    def measure(self, observable: SparsePauli, hint: SparsePauli | None = None) -> None: ...

    def allocate_random_bit(self) -> int:
        """Allocate a random outcome bit, returns its index."""
        ...
    def reserve_qubits(self, new_qubit_capacity: int) -> None:
        """Pre-allocate capacity for qubits to avoid reallocation."""
        ...
    def reserve_outcomes(self, new_outcome_capacity: int, new_random_outcome_capacity: int) -> None:
        """Pre-allocate capacity for measurement outcomes."""
        ...

    def is_stabilizer(self, observable: SparsePauli, ignore_sign: bool = False) -> bool:
        """Check if an observable is in the stabilizer group of current state.
        
        Args:
            observable: Pauli to check.
            ignore_sign: If True, check if ±observable is a stabilizer.
        
        Returns:
            True if observable stabilizes the state.
        """
        ...


@final
class OutcomeCompleteSimulation:
    """Stabilizer simulation with outcome dependence tracking.
    
    Tracks how measurement outcomes depend on random bits and previous outcomes.
    Useful for analyzing measurement patterns and statistical properties.
    
    Examples:
        >>> sim = OutcomeCompleteSimulation(2)
        >>> sim.apply_unitary(UnitaryOpcode.Hadamard, [0])
        >>> sim.measure(SparsePauli("Z_0"))
        >>> sim.outcome_count
        1
    """
    def __init__(self, num_qubits: int = 0) -> None:
        """Create a simulation with the specified number of qubits."""
        ...

    @property
    def qubit_count(self) -> int: ...
    @property
    def qubit_capacity(self) -> int: ...
    @property
    def outcome_count(self) -> int: ...
    @property
    def outcome_capacity(self) -> int: ...
    @property
    def random_outcome_count(self) -> int: ...
    @property
    def random_outcome_capacity(self) -> int: ...
    @property
    def random_bit_count(self) -> int: ...

    def apply_unitary(self, unitary_op: UnitaryOpcode, support: Sequence[int]) -> None: ...
    def apply_pauli_exp(self, observable: SparsePauli) -> None: ...
    def apply_pauli(self, observable: SparsePauli, controlled_by: SparsePauli | None = None) -> None: ...
    def apply_conditional_pauli(
        self, observable: SparsePauli, outcomes: Sequence[int], parity: bool = True,
    ) -> None: ...
    def apply_permutation(self, permutation: Sequence[int], supported_by: Sequence[int] | None = None) -> None: ...
    def apply_clifford(self, clifford: CliffordUnitary, supported_by: Sequence[int] | None = None) -> None: ...
    def measure(self, observable: SparsePauli, hint: SparsePauli | None = None) -> None: ...

    def allocate_random_bit(self) -> int: ...
    def reserve_qubits(self, new_qubit_capacity: int) -> None: ...
    def reserve_outcomes(self, new_outcome_capacity: int, new_random_outcome_capacity: int) -> None: ...

    def is_stabilizer(self, observable: SparsePauli, ignore_sign: bool = False, sign_parity: Sequence[int] = ()) -> bool: ...

    @staticmethod
    def with_capacity(
        qubit_count: int, outcome_count: int, random_outcome_count: int
    ) -> "OutcomeCompleteSimulation": ...

    @property
    def random_outcome_indicator(self) -> BitVector: ...
    @property
    def clifford(self) -> CliffordUnitary: ...
    @property
    def sign_matrix(self) -> BitMatrix: ...
    @property
    def outcome_matrix(self) -> BitMatrix: ...
    @property
    def outcome_shift(self) -> BitVector: ...


@final
class OutcomeFreeSimulation:
    """Stabilizer simulation without outcome tracking.
    
    More efficient when outcome values don't matter, only the final state.
    Suitable for analyzing stabilizer groups without measurement dependence.
    
    Examples:
        >>> sim = OutcomeFreeSimulation(2)
        >>> sim.apply_unitary(UnitaryOpcode.CNOT, [0, 1])
        >>> sim.clifford.is_identity
        False
    """
    def __init__(self, num_qubits: int = 0) -> None:
        """Create a simulation with the specified number of qubits."""
        ...

    @property
    def qubit_count(self) -> int: ...
    @property
    def qubit_capacity(self) -> int: ...
    @property
    def outcome_count(self) -> int: ...
    @property
    def outcome_capacity(self) -> int: ...
    @property
    def random_outcome_count(self) -> int: ...
    @property
    def random_outcome_capacity(self) -> int: ...
    @property
    def random_bit_count(self) -> int: ...

    def apply_unitary(self, unitary_op: UnitaryOpcode, support: Sequence[int]) -> None: ...
    def apply_pauli_exp(self, observable: SparsePauli) -> None: ...
    def apply_pauli(self, observable: SparsePauli, controlled_by: SparsePauli | None = None) -> None: ...
    def apply_conditional_pauli(
        self, observable: SparsePauli, outcomes: Sequence[int], parity: bool = True,
    ) -> None: ...
    def apply_permutation(self, permutation: Sequence[int], supported_by: Sequence[int] | None = None) -> None: ...
    def apply_clifford(self, clifford: CliffordUnitary, supported_by: Sequence[int] | None = None) -> None: ...
    def measure(self, observable: SparsePauli, hint: SparsePauli | None = None) -> None: ...

    def allocate_random_bit(self) -> int: ...
    def reserve_qubits(self, new_qubit_capacity: int) -> None: ...
    def reserve_outcomes(self, new_outcome_capacity: int, new_random_outcome_capacity: int) -> None: ...

    def is_stabilizer(self, observable: SparsePauli, ignore_sign: bool = False, sign_parity: Sequence[int] = ()) -> bool: ...

    @staticmethod
    def with_capacity(
        qubit_count: int, outcome_count: int, random_outcome_count: int
    ) -> "OutcomeFreeSimulation": ...

    @property
    def random_outcome_indicator(self) -> BitVector: ...
    @property
    def clifford(self) -> CliffordUnitary: ...


@final
class OutcomeSpecificSimulation:
    """Stabilizer simulation with concrete outcome values.
    
    Stores specific measurement outcomes, allowing trajectory-based simulation.
    Best for simulating individual runs with deterministic or sampled outcomes.
    
    Examples:
        >>> sim = OutcomeSpecificSimulation(2)
        >>> sim.apply_unitary(UnitaryOpcode.Hadamard, [0])
        >>> sim.measure(SparsePauli("Z_0"))
        >>> sim.outcome_vector.weight  # One measurement recorded
        1
    """
    def __init__(self, num_qubits: int = 0) -> None:
        """Create a simulation with the specified number of qubits."""
        ...

    @property
    def qubit_count(self) -> int: ...
    @property
    def qubit_capacity(self) -> int: ...
    @property
    def outcome_count(self) -> int: ...
    @property
    def outcome_capacity(self) -> int: ...
    @property
    def random_outcome_count(self) -> int: ...
    @property
    def random_outcome_capacity(self) -> int: ...
    @property
    def random_bit_count(self) -> int: ...

    def apply_unitary(self, unitary_op: UnitaryOpcode, support: Sequence[int]) -> None: ...
    def apply_pauli_exp(self, observable: SparsePauli) -> None: ...
    def apply_pauli(self, observable: SparsePauli, controlled_by: SparsePauli | None = None) -> None: ...
    def apply_conditional_pauli(
        self, observable: SparsePauli, outcomes: Sequence[int], parity: bool = True,
    ) -> None: ...
    def apply_permutation(self, permutation: Sequence[int], supported_by: Sequence[int] | None = None) -> None: ...
    def apply_clifford(self, clifford: CliffordUnitary, supported_by: Sequence[int] | None = None) -> None: ...
    def measure(self, observable: SparsePauli, hint: SparsePauli | None = None) -> None: ...

    def allocate_random_bit(self) -> int: ...
    def reserve_qubits(self, new_qubit_capacity: int) -> None: ...
    def reserve_outcomes(self, new_outcome_capacity: int, new_random_outcome_capacity: int) -> None: ...

    def is_stabilizer(self, observable: SparsePauli, ignore_sign: bool = False, sign_parity: Sequence[int] = ()) -> bool: ...

    @staticmethod
    def with_capacity(
        qubit_count: int, outcome_count: int, random_outcome_count: int
    ) -> "OutcomeSpecificSimulation": ...

    @property
    def random_outcome_indicator(self) -> BitVector: ...
    @property
    def outcome_vector(self) -> BitVector: ...


@final
class OutcomeCondition:
    """Condition for applying noise based on measurement outcomes."""
    def __init__(self, outcomes: Sequence[int], parity: bool = True) -> None: ...
    @property
    def outcomes(self) -> list[int]: ...
    @property
    def parity(self) -> bool: ...
    def __repr__(self) -> str: ...


@final
class PauliDistribution:
    """Distribution over Paulis for sampling faults."""
    @staticmethod
    def depolarizing(qubits: Sequence[int]) -> "PauliDistribution":
        """Uniform over all non-identity Paulis on the given qubits."""
        ...
    @staticmethod
    def single(pauli: SparsePauli) -> "PauliDistribution":
        """Single deterministic Pauli."""
        ...
    @staticmethod
    def uniform(paulis: Sequence[SparsePauli]) -> "PauliDistribution":
        """Uniform distribution over an explicit list of Paulis."""
        ...
    @staticmethod
    def weighted(pairs: Sequence[tuple[SparsePauli, float]]) -> "PauliDistribution":
        """Weighted distribution from (Pauli, weight) pairs."""
        ...
    @property
    def elements(self) -> list[tuple[SparsePauli, float]]:
        """All elements as (SparsePauli, probability) pairs."""
        ...
    def __repr__(self) -> str: ...


@final
class PauliFault:
    """A fault specification describing a noise source."""
    def __init__(
        self,
        probability: float,
        distribution: PauliDistribution,
        correlation_id: int | None = None,
        condition: OutcomeCondition | None = None,
    ) -> None: ...
    @staticmethod
    def depolarizing(probability: float, qubits: Sequence[int]) -> "PauliFault":
        """Create a simple depolarizing noise on the given qubits."""
        ...
    @property
    def probability(self) -> float: ...
    @property
    def distribution(self) -> PauliDistribution: ...
    @property
    def correlation_id(self) -> int | None: ...
    @property
    def condition(self) -> OutcomeCondition | None: ...
    def __repr__(self) -> str: ...


@final
class FaultySimulation:
    """Frame-based noisy simulation with circuit-builder interface.
    
    Implements the StabilizerSimulation protocol: call gate methods to build
    a circuit, then call sample() to get noisy outcomes.
    
    Example:
        sim = FaultySimulation()
        sim.apply_unitary(UnitaryOpcode.Hadamard, [0])
        sim.apply_unitary(UnitaryOpcode.ControlledX, [0, 1])
        sim.measure(SparsePauli("ZI"))
        sim.measure(SparsePauli("IZ"))
        sim.apply_fault(PauliFault.depolarizing(0.01, [0, 1]))
        outcomes = sim.sample(1000)
    """
    def __init__(
        self,
        qubit_count: int | None = None,
        outcome_count: int | None = None,
        instruction_count: int | None = None,
    ) -> None:
        """Create a new simulation.
        
        Args:
            qubit_count: Expected number of qubits (optional, for pre-allocation).
            outcome_count: Expected number of measurement outcomes (optional).
            instruction_count: Expected number of instructions (optional).
        
        Pre-allocating capacity can improve performance for large circuits.
        """
        ...
    
    # Properties
    @property
    def qubit_count(self) -> int: ...
    @property
    def outcome_count(self) -> int: ...
    @property
    def fault_count(self) -> int: ...
    
    # Gate methods (StabilizerSimulation protocol)
    def apply_unitary(self, opcode: UnitaryOpcode, qubits: Sequence[int]) -> None: ...
    def apply_clifford(self, clifford: CliffordUnitary, qubits: Sequence[int] | None = None) -> None: ...
    def apply_pauli(self, pauli: SparsePauli, controlled_by: SparsePauli | None = None) -> None: ...
    def apply_pauli_exp(self, pauli: SparsePauli) -> None: ...
    def apply_permutation(self, permutation: Sequence[int], qubits: Sequence[int] | None = None) -> None: ...
    def apply_conditional_pauli(self, pauli: SparsePauli, outcomes: Sequence[int], parity: bool = True) -> None: ...
    def measure(self, observable: SparsePauli, hint: SparsePauli | None = None) -> int:
        """Measure an observable, returning the outcome index."""
        ...
    def allocate_random_bit(self) -> int:
        """Allocate a random bit, returning the outcome index."""
        ...
    
    # Noise methods
    def apply_fault(self, fault: PauliFault) -> None:
        """Add a fault (noise) instruction."""
        ...
    
    # Sampling
    def sample(self, shots: int, seed: int | None = None) -> BitMatrix: ...
    def __repr__(self) -> str: ...