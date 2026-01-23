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
        >>> UnitaryOpcode.ControlledX  # CNOT gate
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
        DensePauli("ZZZ")
    """

    def __init__(self, characters: str = "") -> None:
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
    def identity(size: int) -> "DensePauli":
        """Create the identity operator on `size` qubits."""
        ...

    @staticmethod
    def x(index: int, size: int) -> "DensePauli":
        """Create an X operator at position `index` on `size` qubits."""
        ...

    @staticmethod
    def y(index: int, size: int) -> "DensePauli":
        """Create a Y operator at position `index` on `size` qubits."""
        ...

    @staticmethod
    def z(index: int, size: int) -> "DensePauli":
        """Create a Z operator at position `index` on `size` qubits."""
        ...

    @staticmethod
    def from_sparse(pauli: "SparsePauli", size: int) -> "DensePauli":
        """Convert a SparsePauli to DensePauli on `size` qubits."""
        ...

    @property
    def exponent(self) -> Exponent:
        """The value of `exponent`, when `self` is written in the form e**(iπ * exponent / 4) XᵃZᵇ."""
        ...

    @property
    def phase(self) -> complex:
        """The complex phase of `self` when written in tensor product form e**(iπθ) P₁⊗P₂..., i.e., one of {1, i, -1, -i}."""
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
        >>> p = SparsePauli("X2 Z5")  # X on qubit 2, Z on qubit 5
        >>> p.support
        [2, 5]
        >>> SparsePauli({2: "X", 5: "Z"})  # Dict constructor
    """

    @overload
    def __init__(self, characters: str = "") -> None:
        """Create from string like \"X2 Z5\" or \"X_2 Z_5\".

        Supports formats: \"X0\", \"X_0\", \"X₀\", \"X0 Z5\", etc.
        """
        ...

    @overload
    def __init__(
        self, characters: dict[int, PauliCharacter], exponent: Exponent = 0
    ) -> None:
        """Create from dict mapping qubit indices to Pauli characters.

        Args:
            characters: Dict like {0: \"X\", 3: \"Z\"}.
            exponent: Phase exponent (0, 1, 2, 3 for phases 1, i, -1, -i).
        """
        ...

    @staticmethod
    def identity() -> "SparsePauli":
        """Create the identity operator."""
        ...

    @staticmethod
    def x(index: int) -> "SparsePauli":
        """Create an X operator at the given qubit index."""
        ...

    @staticmethod
    def y(index: int) -> "SparsePauli":
        """Create a Y operator at the given qubit index."""
        ...

    @staticmethod
    def z(index: int) -> "SparsePauli":
        """Create a Z operator at the given qubit index."""
        ...

    @staticmethod
    def from_dense(dense_pauli: DensePauli) -> "SparsePauli":
        """Convert a DensePauli to SparsePauli (drops trailing identities)."""
        ...

    @property
    def exponent(self) -> Exponent:
        """Phase exponent where phase = exp(iπ*exponent/4)."""
        ...

    @property
    def phase(self) -> complex:
        """Complex phase (one of 1, i, -1, -i)."""
        ...

    @property
    def support(self) -> list[int]:
        """List of qubit indices with non-identity operators."""
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
        >>> SparsePauli("X_0 X_1") in group
        True
    """

    def __init__(
        self, generators: Iterable[SparsePauli], all_commute: Optional[bool] = None
    ) -> None:
        """Create a Pauli group from generators.

        Args:
            generators: Iterable of SparsePauli generators.
            all_commute: Hint whether all generators commute (optional optimization).
        """
        ...

    def factorization_of(self, element: SparsePauli) -> Optional[list[SparsePauli]]: ...
    def factorizations_of(
        self, elements: Iterable[SparsePauli]
    ) -> list[Optional[list[SparsePauli]]]: ...
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

def centralizer_of(
    group: PauliGroup, supported_by: Optional[Iterable[int]] = None
) -> PauliGroup:
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
        >>> cnot = CliffordUnitary.from_name("ControlledX", [0, 1], 2)
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
    def from_name(
        name: str, operands: Sequence[int], qubit_count: int
    ) -> "CliffordUnitary":
        """Create a named gate.

        Args:
            name: Gate name (e.g., \"Hadamard\", \"ControlledX\", \"SqrtZ\").
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

def is_diagonal_resource_encoder(
    clifford: CliffordUnitary, axis: Literal["X", "Z"]
) -> bool:
    """Check if a Clifford encodes a diagonal resource state."""
    ...

def unitary_from_diagonal_resource_state(
    clifford: CliffordUnitary, axis: Literal["X", "Z"]
) -> "CliffordUnitary" | None:
    """Extract unitary from a diagonal resource state encoder."""
    ...

def split_qubit_cliffords_and_css(
    clifford: CliffordUnitary,
) -> tuple["CliffordUnitary", "CliffordUnitary"] | None:
    """Split into single-qubit Cliffords and CSS components."""
    ...

def split_phased_css(
    clifford: CliffordUnitary,
) -> tuple["CliffordUnitary", "CliffordUnitary"] | None:
    """Split into phased CSS components."""
    ...

def encoding_clifford_of(
    generators: Sequence[SparsePauli | DensePauli], qubit_count: int
) -> "CliffordUnitary":
    """Construct encoding Clifford from stabilizer generators.

    Args:
        generators: Stabilizer generators.
        qubit_count: Total number of qubits.

    Returns:
        Clifford unitary that maps logical Paulis to given generators.
    """
    ...

class StabilizerSimulation(Protocol):
    """Protocol for stabilizer simulation.

    Defines the common interface implemented by OutcomeCompleteSimulation,
    OutcomeFreeSimulation, OutcomeSpecificSimulation, and FaultySimulation.
    """

    @property
    def qubit_count(self) -> int:
        """Maximum number of qubits in the simulation."""
        ...

    @property
    def qubit_capacity(self) -> int:
        """The number of qubits allowed before reallocation is necessary."""
        ...

    @property
    def outcome_count(self) -> int:
        """Number of measurement outcomes recorded."""
        ...

    @property
    def outcome_capacity(self) -> int:
        """The number of measurement outcomes allowed before reallocation is necessary."""
        ...

    @property
    def random_outcome_count(self) -> int:
        """Number of random outcome bits."""
        ...

    @property
    def random_outcome_capacity(self) -> int:
        """The number of random outcome bits allowed before reallocation is necessary."""
        ...

    @property
    def random_bit_count(self) -> int:
        """Number of random bits involved in the simulation, 
        including both random outcomes and caller supplied random bits."""
        ...

    def apply_unitary(self, unitary_op: UnitaryOpcode, support: Sequence[int]) -> None:
        """Apply a Clifford unitary to specified qubits.

        Args:
            unitary_op: Gate opcode (e.g., UnitaryOpcode.Hadamard, UnitaryOpcode.ControlledX).
            support: Qubit indices where the gate acts.
        """
        ...

    def apply_pauli_exp(self, observable: SparsePauli) -> None:
        """Apply exp(iπ/4 * P) for a Pauli observable P.

        Args:
            observable: Pauli operator to exponentiate.
        """
        ...

    def apply_pauli(
        self, observable: SparsePauli, controlled_by: SparsePauli | None = None
    ) -> None:
        """Apply a Pauli operator, optionally controlled by another Pauli.

        Args:
            observable: Pauli operator to apply.
            controlled_by: Optional control Pauli (gate applies when control eigenvalue is +1).
        """
        ...

    def apply_conditional_pauli(
        self,
        observable: SparsePauli,
        outcomes: Sequence[int],
        parity: bool = True,
    ) -> None:
        """Apply a Pauli conditioned on measurement outcome parity.

        Args:
            observable: Pauli operator to apply.
            outcomes: Measurement outcome indices to check.
            parity: If True, apply when XOR of outcomes is 1; if False, when 0.
        """
        ...

    def apply_permutation(
        self, permutation: Sequence[int], supported_by: Sequence[int] | None = None
    ) -> None:
        """Apply a qubit permutation.

        Args:
            permutation: Mapping where qubit i goes to position permutation[i].
            supported_by: Qubit indices to permute (None means all qubits).
        """
        ...

    def apply_clifford(
        self, clifford: CliffordUnitary, supported_by: Sequence[int] | None = None
    ) -> None:
        """Apply an arbitrary Clifford unitary.

        Args:
            clifford: Clifford unitary to apply.
            supported_by: Qubit indices where the Clifford acts (None infers from clifford.qubit_count).
        """
        ...

    def measure(self, observable: SparsePauli, hint: SparsePauli | None = None) -> None:
        """Measure a Pauli observable, recording the outcome.

        Args:
            observable: Pauli observable to measure.
            hint: Optional anticommuting Pauli to guide simulation (performance optimization).
        """
        ...

    def allocate_random_bit(self) -> int:
        """Allocate a random outcome bit, returns its index."""
        ...

    def reserve_qubits(self, new_qubit_capacity: int) -> None:
        """Pre-allocate capacity for qubits to avoid reallocation."""
        ...

    def reserve_outcomes(
        self, new_outcome_capacity: int, new_random_outcome_capacity: int
    ) -> None:
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
    """Asymptotically efficient stabilizer simulation tracking all measurement outcomes.

    Instead of running separate simulations for each possible measurement outcome,
    this simulator tracks all 2^n_random outcome branches simultaneously, where n_random
    is the number of random measurements. This provides an asymptotic improvement over
    outcome-specific simulation for many use cases.

    Use Cases:
    - **Exhaustive enumeration**: Compute quantities over all possible outcomes
    - **Exact probability distributions**: Calculate measurement statistics without sampling
    - **Circuit verification**: Analyze complete behavior across all measurement branches
    - **Outcome codes**: Study encoding/decoding that depends on measurement outcomes

    Performance:
    - Complexity: O(n_gates × n_qubits²) worst-case, like other simulators
    - Key advantage: Simulate once, then sample any number of shots efficiently
    - Compared to OutcomeSpecific: Saves a factor of n_random when collecting many samples
    - Space: O(n_qubits² + n_random²) for sign and outcome matrices

    The simulation cost is linear in n_random, not exponential. The 2^n_random outcomes
    are represented compactly and can be sampled efficiently.

    Examples:
        >>> sim = OutcomeCompleteSimulation(3)
        >>> sim.apply_unitary(UnitaryOpcode.Hadamard, [0])
        >>> sim.apply_unitary(UnitaryOpcode.ControlledX, [0, 1])
        >>> sim.measure(SparsePauli("Z_0"))
        >>> # All 2^n_random branches tracked without separate simulation runs
        >>> num_branches = 1 << sim.random_outcome_count
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
    def apply_unitary(
        self, unitary_op: UnitaryOpcode, support: Sequence[int]
    ) -> None: ...
    def apply_pauli_exp(self, observable: SparsePauli) -> None: ...
    def apply_pauli(
        self, observable: SparsePauli, controlled_by: SparsePauli | None = None
    ) -> None: ...
    def apply_conditional_pauli(
        self,
        observable: SparsePauli,
        outcomes: Sequence[int],
        parity: bool = True,
    ) -> None: ...
    def apply_permutation(
        self, permutation: Sequence[int], supported_by: Sequence[int] | None = None
    ) -> None: ...
    def apply_clifford(
        self, clifford: CliffordUnitary, supported_by: Sequence[int] | None = None
    ) -> None: ...
    def measure(
        self, observable: SparsePauli, hint: SparsePauli | None = None
    ) -> None: ...
    def allocate_random_bit(self) -> int: ...
    def reserve_qubits(self, new_qubit_capacity: int) -> None: ...
    def reserve_outcomes(
        self, new_outcome_capacity: int, new_random_outcome_capacity: int
    ) -> None: ...
    def is_stabilizer(
        self,
        observable: SparsePauli,
        ignore_sign: bool = False,
        sign_parity: Sequence[int] = (),
    ) -> bool:
        """Check if an observable is a stabilizer of the current state.

        Args:
            observable: Pauli to check.
            ignore_sign: If True, check if ±observable is a stabilizer.
            sign_parity: Outcome indices affecting the sign (for outcome-dependent stabilizers).

        Returns:
            True if observable stabilizes the state.
        """
        ...

    @staticmethod
    def with_capacity(
        qubit_count: int, outcome_count: int, random_outcome_count: int
    ) -> "OutcomeCompleteSimulation":
        """Create simulation with pre-allocated capacity.

        Args:
            qubit_count: Initial qubit capacity.
            outcome_count: Initial outcome capacity.
            random_outcome_count: Initial random outcome capacity.

        Returns:
            New simulation with reserved capacity to avoid reallocations.
        """
        ...

    @property
    def random_outcome_indicator(self) -> BitVector:
        """Indicator of which outcomes are random (vs deterministic)."""
        ...

    @property
    def clifford(self) -> CliffordUnitary:
        """Clifford unitary encoding the current stabilizer state."""
        ...

    @property
    def sign_matrix(self) -> BitMatrix:
        """Sign matrix A encoding how Pauli signs depend on random outcomes.
        
        The sign of stabilizer generator Z_i is determined by the sign parity:
        
            sign_parity_i = A[i, :] · r
        
        where r is the vector of random bit assignments (0 or 1 for each random outcome),
        and · denotes the dot product over GF(2) (XOR). If sign_parity_i = 1, the
        stabilizer has a minus sign; if 0, it's positive.
        
        Shape: (qubit_count, random_outcome_count)
        """
        ...

    @property
    def outcome_matrix(self) -> BitMatrix:
        """Outcome matrix M encoding all 2^k measurement branches.
        
        The value of measurement outcome i is computed from random bit assignments r via:
        
            outcome_i = (M[i, :] · r) ⊕ outcome_shift[i]
        
        where · denotes the dot product over GF(2) (XOR), and ⊕ is XOR. Equivalently,
        for all outcomes as a vector v:
        
            v = M · r ⊕ v_0
        
        where v_0 is the outcome_shift vector. The matrix M is maintained in column-reduced
        form with pivot columns corresponding to random outcome positions.
        
        Shape: (outcome_count, random_outcome_count)
        """
        ...

    @property
    def outcome_shift(self) -> BitVector:
        """Outcome shift vector v_0 representing deterministic outcome contributions.
        
        Combined with outcome_matrix M and random bits r, determines outcome values:
        
            v = M · r ⊕ v_0
        
        For deterministic outcomes (where the corresponding row in M is all zeros),
        outcome_shift directly gives the outcome value. For random outcomes, this
        provides the base value that gets XORed with the linear combination of random bits.
        
        Length: outcome_count
        """
        ...

@final
class OutcomeFreeSimulation:
    """Stabilizer simulation without tracking specific measurement outcomes.

    This simulator tracks the quantum state evolution through measurements without
    committing to specific outcome values. Measurements update the stabilizer state
    modulo Paulis, but outcomes remain unspecified, permitting the fastest simulation.

    The state is represented purely by a Clifford unitary (modulo Paulis) that gets
    updated after each measurement without branching or recording outcome values.

    Use Cases:
    - **Circuit validation**: Verify stabilizer evolution, up to signs

    Performance:
    - Time: O(n_gates × n_qubits²)
    - Space: O(n_qubits²)
    - Most lightweight: No sign tracking overhead

    Examples:
        >>> sim = OutcomeFreeSimulation(3)
        >>> sim.apply_unitary(UnitaryOpcode.Hadamard, [0])
        >>> sim.apply_unitary(UnitaryOpcode.ControlledX, [0, 1])
        >>> sim.measure(SparsePauli("Z_0"))
        >>> # Query stabilizers without caring about the outcome value
        >>> sim.is_stabilizer(SparsePauli("Z_0"))
        True
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
    def apply_unitary(
        self, unitary_op: UnitaryOpcode, support: Sequence[int]
    ) -> None: ...
    def apply_pauli_exp(self, observable: SparsePauli) -> None: ...
    def apply_pauli(
        self, observable: SparsePauli, controlled_by: SparsePauli | None = None
    ) -> None: ...
    def apply_conditional_pauli(
        self,
        observable: SparsePauli,
        outcomes: Sequence[int],
        parity: bool = True,
    ) -> None: ...
    def apply_permutation(
        self, permutation: Sequence[int], supported_by: Sequence[int] | None = None
    ) -> None: ...
    def apply_clifford(
        self, clifford: CliffordUnitary, supported_by: Sequence[int] | None = None
    ) -> None: ...
    def measure(
        self, observable: SparsePauli, hint: SparsePauli | None = None
    ) -> None: ...
    def allocate_random_bit(self) -> int: ...
    def reserve_qubits(self, new_qubit_capacity: int) -> None: ...
    def reserve_outcomes(
        self, new_outcome_capacity: int, new_random_outcome_capacity: int
    ) -> None: ...
    def is_stabilizer(
        self,
        observable: SparsePauli,
        ignore_sign: bool = False,
        sign_parity: Sequence[int] = (),
    ) -> bool:
        """Check if an observable is a stabilizer of the current state.

        Args:
            observable: Pauli to check.
            ignore_sign: If True, check if ±observable is a stabilizer.
            sign_parity: Ignored in outcome-free mode.

        Returns:
            True if observable stabilizes the state (modulo Pauli operators).
        """
        ...

    @staticmethod
    def with_capacity(
        qubit_count: int, outcome_count: int, random_outcome_count: int
    ) -> "OutcomeFreeSimulation":
        """Create simulation with pre-allocated capacity.

        Args:
            qubit_count: Initial qubit capacity.
            outcome_count: Initial outcome capacity.
            random_outcome_count: Initial random outcome capacity.

        Returns:
            New simulation with reserved capacity to avoid reallocations.
        """
        ...

    @property
    def random_outcome_indicator(self) -> BitVector:
        """Indicator of which outcomes are random (vs deterministic)."""
        ...

    @property
    def clifford(self) -> CliffordUnitary:
        """Clifford unitary encoding the current stabilizer state (modulo Paulis)."""
        ...

@final
class OutcomeSpecificSimulation:
    """Traditional stabilizer simulation with random measurement outcomes.

    This simulator draws random measurement outcomes as needed during simulation,
    representing a single execution path through the quantum circuit. Each measurement
    with a random outcome is sampled and recorded, allowing adaptive circuits and
    noise injection based on concrete outcome values.

    Use Cases:
    - **Monte Carlo sampling**: Run many independent shots to estimate error rates
    - **Adaptive circuits**: Runtime measurement outcomes determine subsequent gates
    - **Dynamic noise injection**: Insert novel noise models based on circuit state
    - **Debugging circuits**: Trace specific execution paths with concrete outcomes

    Performance:
    - Complexity: O(n_gates × n_qubits²) worst-case per shot
    - Best for: Few shots or adaptive circuits where next gates depend on outcomes
    - Compared to OutcomeComplete: More efficient when shots << n_random
    - Space: O(n_qubits² + n_measurements)

    Examples:
        >>> # Run multiple shots to collect outcome statistics
        >>> for _ in range(10):
        ...     sim = OutcomeSpecificSimulation(2)
        ...     sim.apply_unitary(UnitaryOpcode.Hadamard, [0])
        ...     sim.apply_unitary(UnitaryOpcode.ControlledX, [0, 1])
        ...     sim.measure(SparsePauli("Z_0"))
        ...     value = sim.outcome_vector[0]  # Access concrete outcome
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
    def apply_unitary(
        self, unitary_op: UnitaryOpcode, support: Sequence[int]
    ) -> None: ...
    def apply_pauli_exp(self, observable: SparsePauli) -> None: ...
    def apply_pauli(
        self, observable: SparsePauli, controlled_by: SparsePauli | None = None
    ) -> None: ...
    def apply_conditional_pauli(
        self,
        observable: SparsePauli,
        outcomes: Sequence[int],
        parity: bool = True,
    ) -> None: ...
    def apply_permutation(
        self, permutation: Sequence[int], supported_by: Sequence[int] | None = None
    ) -> None: ...
    def apply_clifford(
        self, clifford: CliffordUnitary, supported_by: Sequence[int] | None = None
    ) -> None: ...
    def measure(
        self, observable: SparsePauli, hint: SparsePauli | None = None
    ) -> None: ...
    def allocate_random_bit(self) -> int: ...
    def reserve_qubits(self, new_qubit_capacity: int) -> None: ...
    def reserve_outcomes(
        self, new_outcome_capacity: int, new_random_outcome_capacity: int
    ) -> None: ...
    def is_stabilizer(
        self,
        observable: SparsePauli,
        ignore_sign: bool = False,
        sign_parity: Sequence[int] = (),
    ) -> bool:
        """Check if an observable is a stabilizer of the current state.

        Args:
            observable: Pauli to check.
            ignore_sign: If True, check if ±observable is a stabilizer.
            sign_parity: Ignored in outcome-specific mode.

        Returns:
            True if observable stabilizes the state.
        """
        ...

    @staticmethod
    def with_capacity(
        qubit_count: int, outcome_count: int, random_outcome_count: int
    ) -> "OutcomeSpecificSimulation":
        """Create simulation with pre-allocated capacity.

        Args:
            qubit_count: Initial qubit capacity.
            outcome_count: Initial outcome capacity.
            random_outcome_count: Initial random outcome capacity.

        Returns:
            New simulation with reserved capacity to avoid reallocations.
        """
        ...

    @property
    def random_outcome_indicator(self) -> BitVector:
        """Indicator of which outcomes are random (vs deterministic)."""
        ...

    @property
    def outcome_vector(self) -> BitVector:
        """Concrete measurement outcome values for this trajectory."""
        ...

@final
class OutcomeCondition:
    """Condition for applying noise based on measurement outcomes.

    Specifies that noise should only be applied when a particular parity of
    measurement outcomes is satisfied.

    Args:
        outcomes: Indices of measurement outcomes to check.
        parity: If True, apply when XOR of outcomes is 1 (odd); if False, when 0 (even).

    Examples:
        >>> # Apply noise only when outcome 0 XOR outcome 1 = 1
        >>> condition = OutcomeCondition([0, 1], parity=True)
    """

    def __init__(self, outcomes: Sequence[int], parity: bool = True) -> None: ...
    @property
    def outcomes(self) -> list[int]:
        """Measurement outcome indices to check."""
        ...

    @property
    def parity(self) -> bool:
        """Target parity: True for odd (1), False for even (0)."""
        ...

    def __repr__(self) -> str: ...

@final
class PauliDistribution:
    """Probability distribution over Pauli operators.

    Defines a random variable P taking values in a set of Pauli operators {P_i}
    with probabilities {q_i}, where sum(q_i) = 1.

    While typically used to describe the error term of a noisy channel (the
    distribution of a PauliFault), this type itself is strictly a distribution
    definition. It does not encode the probability of the channel acting.

    Variants:
    - **Single**: Distribution with a single element (q_0 = 1)
    - **Depolarizing**: Uniform over all 4^n - 1 non-identity Paulis on n qubits
    - **Uniform**: Uniform over an explicit list of Paulis
    - **Weighted**: Arbitrary distribution defined by (P_i, q_i) pairs

    Examples:
        >>> # Single deterministic Pauli
        >>> dist = PauliDistribution.single(SparsePauli("X0"))
        >>>
        >>> # Depolarizing noise on qubits 0 and 1
        >>> dist = PauliDistribution.depolarizing([0, 1])
        >>>
        >>> # Custom weighted distribution
        >>> dist = PauliDistribution.weighted([(SparsePauli("X0"), 0.9), (SparsePauli("Z0"), 0.1)])
    """

    @staticmethod
    def depolarizing(qubits: Sequence[int]) -> "PauliDistribution":
        """Uniform over all non-identity Paulis on the given qubits.

        Samples uniformly from all 4^n - 1 non-identity Pauli operators on n qubits.
        Uses fast bit sampling: O(1) space, O(k) time for k qubits.

        Args:
            qubits: Qubit indices for the depolarizing noise.

        Returns:
            Distribution over all non-identity Paulis on these qubits.
        """
        ...

    @staticmethod
    def single(pauli: SparsePauli) -> "PauliDistribution":
        """Single deterministic Pauli (no randomness).

        Args:
            pauli: The deterministic Pauli operator.

        Returns:
            Distribution that always returns this Pauli.
        """
        ...

    @staticmethod
    def uniform(paulis: Sequence[SparsePauli]) -> "PauliDistribution":
        """Uniform distribution over an explicit list of Paulis.

        Args:
            paulis: List of Pauli operators to sample uniformly from.

        Returns:
            Distribution with equal probability for each Pauli.
        """
        ...

    @staticmethod
    def weighted(pairs: Sequence[tuple[SparsePauli, float]]) -> "PauliDistribution":
        """Weighted distribution from (Pauli, weight) pairs.

        Weights are normalized to sum to 1. Uses binary search for efficient sampling.

        Args:
            pairs: Sequence of (Pauli, weight) tuples. Weights must sum to a positive value.

        Returns:
            Distribution with specified relative probabilities.

        Raises:
            AssertionError: If weights don't sum to a positive value.
        """
        ...

    @property
    def elements(self) -> list[tuple[SparsePauli, float]]:
        """All elements as (SparsePauli, probability) pairs.

        For depolarizing noise, this enumerates all 4^n - 1 non-identity Paulis
        with uniform probabilities. May be expensive for large n.
        """
        ...

    def __repr__(self) -> str: ...

@final
class PauliFault:
    """Fault specification describing a noise source.

    A fault combines:
    - A probability that an error occurs
    - A distribution over Pauli operators given that an error occurred
    - Optional correlation ID for modeling correlated errors across space/time
    - Optional condition based on measurement outcomes

    The overall effect is: with probability p, sample a Pauli from the distribution
    and apply it (subject to the optional condition).

    Correlated Faults (correlation_id):
        When multiple PauliFault instructions share the same correlation_id, they are
        treated as a single probabilistic event:
        
        1. **Trigger Coupling**: In any given simulation shot, either *all* faults with
           the same ID trigger, or *none* of them trigger.
        2. **Sample Coupling**: If they trigger, they sample from their distributions using
           the same random index.
        
        This is primarily intended for **time-like correlations**: modeling the same noise source
        affecting a qubit at different points in time. For example, a qubit experiencing a
        slow drift that causes correlated errors at multiple gate locations in the circuit.
        
        Note: For space-like correlations (e.g., crosstalk affecting multiple qubits
        simultaneously), use a single PauliFault with a distribution over multi-qubit Paulis
        rather than correlation_id.
        
        **Constraint**: All faults with the same correlation_id must have the same
        probability and distributions of the same size.
        
        Example:
            >>> # Time-correlated noise: qubit 0 experiences same error type early and late
            >>> dist = PauliDistribution.uniform([SparsePauli("X0"), SparsePauli("Z0")])
            >>> fault_early = PauliFault(0.02, dist, correlation_id=1)
            >>> # ... gates in between ...
            >>> fault_late = PauliFault(0.02, dist, correlation_id=1)
            >>> # Both sample X or both sample Z in each shot

    Conditional Faults (condition):
        When a condition is specified, the fault is only active if the condition is satisfied
        (i.e., the XOR of specified measurement outcomes matches the required parity):
        
        - If condition is **false**: fault is suppressed (probability = 0)
        - If condition is **true**: fault occurs with the specified probability
        
        This enables modeling of outcome-dependent errors.
        
        Example:
            >>> # Apply Z error only when measurements 0 XOR 1 = 1 (odd parity)
            >>> condition = OutcomeCondition([0, 1], parity=True)
            >>> fault = PauliFault(0.05, PauliDistribution.single(SparsePauli("Z0")), condition=condition)

    Examples:
        >>> # Simple depolarizing noise with 1% error rate
        >>> fault = PauliFault.depolarizing(0.01, [0, 1])
        >>>
        >>> # Custom fault with correlation
        >>> dist = PauliDistribution.single(SparsePauli("X0"))
        >>> fault = PauliFault(0.05, dist, correlation_id=1)
        >>>
        >>> # Conditional fault based on measurement outcomes
        >>> condition = OutcomeCondition([0, 1], parity=True)
        >>> fault = PauliFault(0.1, dist, condition=condition)
    """

    def __init__(
        self,
        probability: float,
        distribution: PauliDistribution,
        correlation_id: int | None = None,
        condition: OutcomeCondition | None = None,
    ) -> None:
        """Create a fault specification.

        Args:
            probability: Probability that a fault occurs (0.0 to 1.0).
            distribution: Distribution over Paulis given a fault.
            correlation_id: Optional ID for correlated faults. All faults with the same
                correlation_id will trigger together in each shot and sample the same
                error from their distributions. Must have matching probability and
                distribution sizes.
            condition: Optional condition on measurement outcomes. If specified, the fault
                only occurs when the XOR of the specified outcomes matches the parity.
        """
        ...

    @staticmethod
    def depolarizing(probability: float, qubits: Sequence[int]) -> "PauliFault":
        """Create depolarizing noise on the given qubits.

        Convenience method for uniform noise over all non-identity Paulis.

        Args:
            probability: Probability of a Pauli error (0.0 to 1.0).
            qubits: Qubit indices affected by the noise.

        Returns:
            Fault with depolarizing distribution on these qubits.
        """
        ...

    @property
    def probability(self) -> float:
        """Probability that a fault occurs."""
        ...

    @property
    def distribution(self) -> PauliDistribution:
        """Distribution over Paulis given that a fault occurred."""
        ...

    @property
    def correlation_id(self) -> int | None:
        """Correlation ID for time-correlated faults.
        
        Faults sharing the same correlation_id trigger together in each simulation shot
        and sample using the same random index from their distributions. Primarily used
        to model time-like correlations: the same noise source affecting a location at
        multiple points in the circuit (e.g., slow drifts, memory errors).
        
        Returns None if the fault is uncorrelated (independent from all other faults).
        """
        ...

    @property
    def condition(self) -> OutcomeCondition | None:
        """Optional condition based on measurement outcomes.
        
        When present, the fault is only active when the condition is satisfied (i.e.,
        the XOR of the specified measurement outcomes equals the target parity).
        If the condition is false, the fault is suppressed (acts as if probability=0).
        If the condition is true, the fault occurs with its specified probability.
        
        Returns None if the fault is unconditional (always active).
        """
        ...

    def __repr__(self) -> str: ...

@final
class FaultySimulation:
    """Noisy stabilizer simulation using frame-based error propagation.

    This simulator combines noiseless outcome sampling (OutcomeCompleteSimulation)
    with efficient frame-based error propagation. Noise is represented as Pauli
    errors that propagate through Clifford gates, enabling O(n_gates × n_qubits²)
    complexity for multi-shot noisy simulation.

    Rather than tracking full noisy quantum states, errors are represented as Pauli
    frames that commute through gates. This allows efficient simulation of realistic
    noise models while maintaining the O(n²) scaling of stabilizer simulation.

    Use Cases:
    - **Logical error rates**: Estimate logical error rates via Monte Carlo sampling
    - **Noise characterization**: Study error propagation under different noise models
    - **Decoder validation**: Test decoder performance with realistic noise

    Performance:
    - Complexity: O(n_gates × n_qubits²) worst-case, same as noiseless simulators
    - Frame propagation: Tracks Pauli errors through gates efficiently
    - Sampling cost: O(shots × (n_gates × n_qubits + n_measurements × n_random)) total
    - Space: O(n_qubits² + n_measurements² + shots × n_measurements)

    Noise Models:
    Supports various noise distributions via PauliFault:
    - Single Pauli errors: PauliDistribution.single()
    - Depolarizing noise: PauliFault.depolarizing()
    - Weighted distributions: PauliDistribution.weighted()
    - Conditional errors: Based on measurement outcomes
    - Correlated errors: Via correlation IDs

    Examples:
        >>> sim = FaultySimulation()
        >>> sim.apply_unitary(UnitaryOpcode.Hadamard, [0])
        >>> sim.apply_fault(PauliFault.depolarizing(0.01, [0]))
        >>> sim.apply_unitary(UnitaryOpcode.ControlledX, [0, 1])
        >>> sim.apply_fault(PauliFault.depolarizing(0.01, [0, 1]))
        >>> sim.measure(SparsePauli("Z_0"))
        >>> sim.measure(SparsePauli("Z_1"))
        >>> outcomes = sim.sample(100)  # Sample 100 noisy shots
        >>> outcomes.shape
        (100, 2)
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
    def qubit_count(self) -> int:
        """Current number of qubits in use."""
        ...

    @property
    def outcome_count(self) -> int:
        """Number of measurement outcomes recorded."""
        ...

    @property
    def fault_count(self) -> int:
        """Number of fault (noise) instructions in the circuit."""
        ...
    # Gate methods (StabilizerSimulation protocol)
    def apply_unitary(self, opcode: UnitaryOpcode, qubits: Sequence[int]) -> None: ...
    def apply_clifford(
        self, clifford: CliffordUnitary, qubits: Sequence[int] | None = None
    ) -> None: ...
    def apply_pauli(
        self, pauli: SparsePauli, controlled_by: SparsePauli | None = None
    ) -> None: ...
    def apply_pauli_exp(self, pauli: SparsePauli) -> None: ...
    def apply_permutation(
        self, permutation: Sequence[int], qubits: Sequence[int] | None = None
    ) -> None: ...
    def apply_conditional_pauli(
        self, pauli: SparsePauli, outcomes: Sequence[int], parity: bool = True
    ) -> None: ...
    def measure(self, observable: SparsePauli, hint: SparsePauli | None = None) -> int:
        """Measure an observable, returning the outcome index."""
        ...

    def allocate_random_bit(self) -> int:
        """Allocate a random bit, returning the outcome index."""
        ...
    # Noise methods
    def apply_fault(self, fault: PauliFault) -> None:
        """Add a fault (noise) instruction to the circuit.

        Args:
            fault: Fault specification describing the noise to apply.
        """
        ...
    # Sampling
    def sample(self, shots: int, seed: int | None = None) -> BitMatrix:
        """Sample noisy measurement outcomes.

        Args:
            shots: Number of independent samples to generate.
            seed: Optional random seed for reproducibility.

        Returns:
            BitMatrix with shape (shots, outcome_count) containing measurement outcomes.
            The result is row-major.  Each shot corresponds to a row.
        """
        ...

    def __repr__(self) -> str: ...
