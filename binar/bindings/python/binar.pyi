from typing import Iterable, final, Iterator, Literal, overload

Bits = str | Iterable[bool | Literal[0, 1]]

@final
class BitMatrix:
    """Matrix of bits stored efficiently for linear algebra over GF(2).
    
    BitMatrix provides high-performance operations for binary matrices including
    matrix multiplication, echelonization, kernel computation, and solving linear
    systems over GF(2). Ideal for quantum error correction, coding theory, and
    stabilizer simulation.
    
    Examples:
        >>> m = BitMatrix([[1, 0, 1], [0, 1, 1]])
        >>> m.shape
        (2, 3)
        >>> m @ m.T
        BitMatrix([[0, 1], [1, 0]])
    """
    def __init__(self, rows: Iterable["BitVector" | Bits]) -> None:
        """Create a BitMatrix from an iterable of rows.
        
        Args:
            rows: Iterable of BitVector or bit sequences (strings like "101" or lists of bools).
        
        Examples:
            >>> BitMatrix(["101", "011"])
            >>> BitMatrix([[True, False], [False, True]])
        """
        ...
    @staticmethod
    def identity(dimension: int) -> "BitMatrix":
        """Create an identity matrix of given dimension."""
        ...
    @staticmethod
    def zeros(rows: int, columns: int) -> "BitMatrix":
        """Create a zero matrix with the specified dimensions."""
        ...
    @staticmethod
    def ones(rows: int, columns: int) -> "BitMatrix":
        """Create a matrix filled with ones."""
        ...
    @staticmethod
    def _from_bytes(rows: int, columns: int, data: bytes) -> "BitMatrix": ...
    @property
    def row_count(self) -> int:
        """Number of rows in the matrix."""
        ...
    @property
    def column_count(self) -> int:
        """Number of columns in the matrix."""
        ...
    @property
    def shape(self) -> tuple[int, int]:
        """Matrix dimensions as (rows, columns)."""
        ...
    @property
    def rows(self) -> Iterator["BitVector"]:
        """Iterator over the rows of the matrix."""
        ...
    @property
    def T(self) -> "BitMatrix":
        """Transpose of the matrix."""
        ...
    @property
    def ndim(self) -> int:
        """Number of dimensions (always 2)."""
        ...
    @property
    def size(self) -> int:
        """Total number of elements (row_count * column_count)."""
        ...
    def copy(self) -> "BitMatrix":
        """Create a copy of this matrix."""
        ...
    def reshape(self, rows: int, columns: int) -> None:
        """Reshape the matrix in-place to new dimensions.
        
        Args:
            rows: New number of rows.
            columns: New number of columns.
        
        Raises:
            ValueError: If rows * columns != size.
        """
        ...
    def dot(self, other: "BitMatrix") -> "BitMatrix":
        """Compute matrix product over GF(2) (addition is XOR).
        
        Args:
            other: Right-hand matrix.
        
        Returns:
            Result of self @ other.
        """
        ...
    def submatrix(self, rows: list[int], columns: list[int]) -> "BitMatrix":
        """Extract a submatrix at specified row and column indices.
        
        Args:
            rows: Row indices to include.
            columns: Column indices to include.
        
        Returns:
            New matrix with selected rows and columns.
        """
        ...
    def echelonize(self) -> list[int]:
        """Convert to row echelon form in-place.
        
        Returns:
            List of pivot column indices.
        """
        ...
    def echelonized(self) -> "BitMatrix":
        """Return a copy of this matrix in row echelon form."""
        ...
    def kernel(self) -> "BitMatrix":
        """Compute the kernel (null space) of the matrix.
        
        Returns:
            Matrix whose rows form a basis for the kernel.
        """
        ...
    def __getitem__(self, index: tuple[int, int]) -> bool: ...
    def __setitem__(self, index: tuple[int, int], to: bool) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __add__(self, other: "BitMatrix") -> "BitMatrix": ...
    def __iadd__(self, other: "BitMatrix") -> None: ...
    def __mul__(self, other: "BitMatrix") -> "BitMatrix": ...
    @overload
    def __matmul__(self, other: "BitMatrix") -> "BitMatrix": ...
    @overload
    def __matmul__(self, other: "BitVector") -> "BitVector": ...
    def __xor__(self, other: "BitMatrix") -> "BitMatrix": ...
    def __ixor__(self, other: "BitMatrix") -> None: ...
    def __and__(self, other: "BitMatrix") -> "BitMatrix": ...
    def __iand__(self, other: "BitMatrix") -> None: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def _to_bytes(self) -> bytes: ...

@final
class BitVector:
    """Compact bit vector with efficient bitwise operations.
    
    BitVector stores a sequence of bits with fast operations including XOR,
    AND, OR, and weight computation. Used throughout the library for
    representing measurement outcomes, syndrome patterns, and binary vectors.
    
    Examples:
        >>> v = BitVector("1010")
        >>> v.weight
        2
        >>> v ^ BitVector("1100")
        BitVector("0110")
    """
    def __init__(self, bits: Bits) -> None:
        """Create a BitVector from a bit sequence.
        
        Args:
            bits: String like "1010", or iterable of bools/0/1.
        
        Examples:
            >>> BitVector("101")
            >>> BitVector([True, False, True])
            >>> BitVector([1, 0, 1])
        """
        ...
    @staticmethod
    def zeros(length: int) -> "BitVector":
        """Create a zero vector of given length."""
        ...
    @staticmethod
    def ones(length: int) -> "BitVector":
        """Create a vector of all ones."""
        ...
    @staticmethod
    def _from_bytes(length: int, data: bytes) -> "BitVector": ...
    @property
    def weight(self) -> int:
        """Hamming weight (number of 1 bits)."""
        ...
    @property
    def parity(self) -> bool:
        """Parity of the bit vector (True if odd weight)."""
        ...
    @property
    def is_zero(self) -> bool:
        """True if all bits are zero."""
        ...
    @property
    def support(self) -> list[int]:
        """Indices where bits are set to 1."""
        ...
    def resize(self, new_length: int) -> None:
        """Resize the vector in-place, truncating or zero-padding."""
        ...
    def copy(self) -> "BitVector":
        """Create a copy of this bit vector."""
        ...
    def clear(self) -> None:
        """Set all bits to zero."""
        ...
    def negate_index(self, index: int) -> None:
        """Flip the bit at the given index."""
        ...
    def dot(self, other: "BitVector") -> bool:
        """Inner product over GF(2).
        
        Returns:
            True if the dot product is 1 (odd), False if 0 (even).
        """
        ...
    def and_weight(self, other: "BitVector") -> int:
        """Hamming weight of the bitwise AND."""
        ...
    def or_weight(self, other: "BitVector") -> int:
        """Hamming weight of the bitwise OR."""
        ...
    def __iter__(self) -> Iterator[bool]: ...
    @overload
    def __getitem__(self, index: int) -> bool: ...
    @overload
    def __getitem__(self, index: slice) -> "BitVector": ...
    def __setitem__(self, index: int, to: bool) -> None: ...
    def __len__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...
    def __xor__(self, other: "BitVector") -> "BitVector": ...
    def __ixor__(self, other: "BitVector") -> None: ...
    def __and__(self, other: "BitVector") -> "BitVector": ...
    def __iand__(self, other: "BitVector") -> None: ...
    def __or__(self, other: "BitVector") -> "BitVector": ...
    def __ior__(self, other: "BitVector") -> None: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def _to_bytes(self) -> bytes: ...


def vstack(matrices: Iterable[BitMatrix]) -> BitMatrix:
    """Stack matrices vertically (concatenate rows).
    
    Args:
        matrices: Iterable of BitMatrix with the same number of columns.
    
    Returns:
        New matrix with all input matrices stacked vertically.
    """
    ...

def rank(matrix: BitMatrix) -> int:
    """Compute the rank of the matrix over GF(2)."""
    ...
def null_space(matrix: BitMatrix) -> BitMatrix:
    """Compute the null space (kernel) of the matrix.
    
    Returns:
        Matrix whose rows form a basis for the null space.
    """
    ...
def inv(matrix: BitMatrix) -> BitMatrix:
    """Compute the inverse of a square matrix over GF(2).
    
    Raises:
        ValueError: If the matrix is not invertible.
    """
    ...
def det(matrix: BitMatrix) -> bool:
    """Compute the determinant over GF(2) (True if odd, False if even)."""
    ...
def solve(matrix: BitMatrix, b: BitVector) -> BitVector | None:
    """Solve the linear system Ax = b over GF(2).
    
    Args:
        matrix: Coefficient matrix A.
        b: Right-hand side vector.
    
    Returns:
        Solution vector if one exists, None otherwise.
    """
    ...
