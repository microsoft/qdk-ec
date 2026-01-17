try:
    import numpy as np
    import galois
    HAS_GALOIS = True
except ImportError:
    HAS_GALOIS = False

from binar import BitMatrix


class BitMatrixInitialization:
    params = [[10, 50, 100, 500], [10, 50, 100, 500]]
    param_names = ['rows', 'cols']
    
    def setup(self, rows, cols):
        self.bool_lists = [[bool((i * cols + j) % 2) for j in range(cols)] for i in range(rows)]
        if HAS_GALOIS:
            self.np_array = np.array(self.bool_lists, dtype=np.uint8)
    
    def time_bitmatrix_from_lists(self, rows, cols):
        BitMatrix(self.bool_lists)
    
    def time_galois_from_numpy(self, rows, cols):
        if not HAS_GALOIS:
            raise NotImplementedError("galois not installed")
        galois.GF2(self.np_array)
    
    def time_bitmatrix_zeros(self, rows, cols):
        BitMatrix.zeros(rows, cols)
    
    def time_galois_zeros(self, rows, cols):
        if not HAS_GALOIS:
            raise NotImplementedError("galois not installed")
        galois.GF2.Zeros((rows, cols))
    
    def time_bitmatrix_ones(self, rows, cols):
        BitMatrix.ones(rows, cols)
    
    def time_galois_ones(self, rows, cols):
        if not HAS_GALOIS:
            raise NotImplementedError("galois not installed")
        galois.GF2.Ones((rows, cols))
    
    def time_bitmatrix_identity(self, rows, cols):
        size = min(rows, cols)
        BitMatrix.identity(size)
    
    def time_galois_identity(self, rows, cols):
        if not HAS_GALOIS:
            raise NotImplementedError("galois not installed")
        size = min(rows, cols)
        galois.GF2.Identity(size)


class BitMatrixBinaryOperations:
    params = [[10, 50, 100, 500], [10, 50, 100, 500]]
    param_names = ['rows', 'cols']
    
    def setup(self, rows, cols):
        # Create BitMatrix instances
        self.m1 = BitMatrix([[bool((i * cols + j) % 2) for j in range(cols)] for i in range(rows)])
        self.m2 = BitMatrix([[bool((i * cols + j + 1) % 2) for j in range(cols)] for i in range(rows)])
        
        # Create galois instances
        if HAS_GALOIS:
            self.g1 = galois.GF2([[int((i * cols + j) % 2) for j in range(cols)] for i in range(rows)])
            self.g2 = galois.GF2([[int((i * cols + j + 1) % 2) for j in range(cols)] for i in range(rows)])
    
    def time_bitmatrix_add(self, rows, cols):
        _ = self.m1 + self.m2
    
    def time_bitmatrix_add_inplace(self, rows, cols):
        self.m1 += self.m2
    
    def time_galois_add(self, rows, cols):
        if not HAS_GALOIS:
            raise NotImplementedError("galois not installed")
        _ = self.g1 + self.g2
    
    def time_galois_add_inplace(self, rows, cols):
        if not HAS_GALOIS:
            raise NotImplementedError("galois not installed")
        self.g1 += self.g2
    
    def time_bitmatrix_xor(self, rows, cols):
        _ = self.m1 ^ self.m2
    
    def time_bitmatrix_xor_inplace(self, rows, cols):
        self.m1 ^= self.m2
    
    def time_galois_xor(self, rows, cols):
        if not HAS_GALOIS:
            raise NotImplementedError("galois not installed")
        _ = self.g1 ^ self.g2
    
    def time_galois_xor_inplace(self, rows, cols):
        if not HAS_GALOIS:
            raise NotImplementedError("galois not installed")
        self.g1 ^= self.g2
    
    def time_bitmatrix_and(self, rows, cols):
        _ = self.m1 & self.m2
    
    def time_bitmatrix_and_inplace(self, rows, cols):
        self.m1 &= self.m2
    
    def time_galois_and(self, rows, cols):
        if not HAS_GALOIS:
            raise NotImplementedError("galois not installed")
        _ = self.g1 & self.g2
    
    def time_galois_and_inplace(self, rows, cols):
        if not HAS_GALOIS:
            raise NotImplementedError("galois not installed")
        self.g1 &= self.g2


class BitMatrixMultiplication:
    params = [[10, 50, 100, 200]]
    param_names = ['size']
    
    def setup(self, size):
        # Square matrices for multiplication
        self.m1 = BitMatrix([[bool((i * size + j) % 3 == 0) for j in range(size)] for i in range(size)])
        self.m2 = BitMatrix([[bool((i * size + j + 1) % 3 == 0) for j in range(size)] for i in range(size)])
        
        if HAS_GALOIS:
            self.g1 = galois.GF2([[int((i * size + j) % 3 == 0) for j in range(size)] for i in range(size)])
            self.g2 = galois.GF2([[int((i * size + j + 1) % 3 == 0) for j in range(size)] for i in range(size)])
    
    def time_bitmatrix_matmul(self, size):
        _ = self.m1 @ self.m2
    
    def time_galois_matmul(self, size):
        if not HAS_GALOIS:
            raise NotImplementedError("galois not installed")
        _ = self.g1 @ self.g2


class BitMatrixIndexingAndSlicing:
    params = [[50, 100, 500]]
    param_names = ['size']
    
    def setup(self, size):
        self.m = BitMatrix.zeros(size, size)
        if HAS_GALOIS:
            self.g = galois.GF2.Zeros((size, size))
    
    def time_bitmatrix_getitem(self, size):
        _ = self.m[size // 2, size // 2]
    
    def time_galois_getitem(self, size):
        if not HAS_GALOIS:
            raise NotImplementedError("galois not installed")
        _ = self.g[size // 2, size // 2]
    
    def time_bitmatrix_setitem(self, size):
        self.m[size // 2, size // 2] = True
    
    def time_galois_setitem(self, size):
        if not HAS_GALOIS:
            raise NotImplementedError("galois not installed")
        self.g[size // 2, size // 2] = 1


class BitMatrixTransformations:
    params = [[50, 100, 200]]
    param_names = ['size']
    
    def setup(self, size):
        self.m = BitMatrix([[bool((i * size + j) % 3 == 0) for j in range(size)] for i in range(size)])
        if HAS_GALOIS:
            self.g = galois.GF2([[int((i * size + j) % 3 == 0) for j in range(size)] for i in range(size)])
    
    def time_bitmatrix_transpose(self, size):
        _ = self.m.T
    
    def time_galois_transpose(self, size):
        if not HAS_GALOIS:
            raise NotImplementedError("galois not installed")
        _ = self.g.T
    
    def time_bitmatrix_copy(self, size):
        _ = self.m.copy()
    
    def time_galois_copy(self, size):
        if not HAS_GALOIS:
            raise NotImplementedError("galois not installed")
        _ = self.g.copy()


class BitMatrixLinearAlgebra:
    params = [[10, 50, 100]]
    param_names = ['size']
    
    def setup(self, size):
        # Create matrices with some structure for interesting linear algebra
        self.m = BitMatrix([[bool((i + j) % 3 < 2) for j in range(size)] for i in range(size)])
        
        if HAS_GALOIS:
            self.g = galois.GF2([[int((i + j) % 3 < 2) for j in range(size)] for i in range(size)])
    
    def time_bitmatrix_echelonize(self, size):
        m_copy = self.m.copy()
        m_copy.echelonize()
    
    def time_bitmatrix_echelonized(self, size):
        m_copy = self.m.copy()
        _ = m_copy.echelonized()
    
    def time_galois_row_reduce(self, size):
        if not HAS_GALOIS:
            raise NotImplementedError("galois not installed")
        _ = self.g.row_reduce()
    
    def time_bitmatrix_kernel(self, size):
        m_copy = self.m.copy()
        _ = m_copy.kernel()
    
    def time_galois_null_space(self, size):
        if not HAS_GALOIS:
            raise NotImplementedError("galois not installed")
        _ = self.g.null_space()


class BitMatrixModuleFunctions:
    params = [[10, 50, 100]]
    param_names = ['size']
    
    def setup(self, size):
        from binar import rank, null_space, det
        
        self.m = BitMatrix([[bool((i + j) % 3 < 2) for j in range(size)] for i in range(size)])
        self.rank_func = rank
        self.null_space_func = null_space
        self.det_func = det
        
        if HAS_GALOIS:
            self.g = galois.GF2([[int((i + j) % 3 < 2) for j in range(size)] for i in range(size)])
    
    def time_bitmatrix_rank(self, size):
        _ = self.rank_func(self.m)
    
    def time_galois_rank(self, size):
        if not HAS_GALOIS:
            raise NotImplementedError("galois not installed")
        _ = np.linalg.matrix_rank(self.g)
    
    def time_bitmatrix_null_space(self, size):
        _ = self.null_space_func(self.m)
    
    def time_galois_null_space(self, size):
        if not HAS_GALOIS:
            raise NotImplementedError("galois not installed")
        _ = self.g.null_space()
    
    def time_bitmatrix_det(self, size):
        _ = self.det_func(self.m)
    
    def time_galois_det(self, size):
        if not HAS_GALOIS:
            raise NotImplementedError("galois not installed")
        _ = int(np.linalg.det(self.g))


class BitMatrixMemoryFootprint:
    """Memory benchmarks to compare storage efficiency."""
    params = [[100, 500, 1000]]
    param_names = ['size']
    
    def setup(self, size):
        self.bool_lists = [[bool((i * size + j) % 2) for j in range(size)] for i in range(size)]
    
    def peakmem_bitmatrix_create(self, size):
        _ = BitMatrix(self.bool_lists)
    
    def peakmem_galois_create(self, size):
        if not HAS_GALOIS:
            raise NotImplementedError("galois not installed")
        _ = galois.GF2(self.bool_lists)