try:
    from bitarray import bitarray
    HAS_BITARRAY = True
except ImportError:
    HAS_BITARRAY = False

from binar import BitVector


class BitVectorInitialization:
    params = [[100, 1000, 10000]]
    param_names = ['size']
    
    def setup(self, size):
        self.bool_list = [i % 2 == 0 for i in range(size)]
        self.bool_tuple = tuple(self.bool_list)
        self.string = "".join(str(int(value)) for value in self.bool_list)
    
    def time_bitvector_list_bool(self, size):
        BitVector(self.bool_list)
    
    def time_bitarray_list_bool(self, size):
        if not HAS_BITARRAY:
            raise NotImplementedError("bitarray not installed")
        bitarray(self.bool_list)

    def time_bitvector_tuple_bool(self, size):
        BitVector(self.bool_tuple)
    
    def time_bitarray_tuple_bool(self, size):
        if not HAS_BITARRAY:
            raise NotImplementedError("bitarray not installed")
        bitarray(self.bool_tuple)

    def time_bitvector_string(self, size):
        BitVector(self.string)
    
    def time_bitarray_string(self, size):
        if not HAS_BITARRAY:
            raise NotImplementedError("bitarray not installed")
        bitarray(self.string)

    def time_bitvector_iter_bool(self, size):
        BitVector(iter(self.bool_list))
    
    def time_bitarray_iter_bool(self, size):
        if not HAS_BITARRAY:
            raise NotImplementedError("bitarray not installed")
        bitarray(iter(self.bool_list))


class BitVectorBinaryOperations:
    params = [[100, 1000, 10000, 100000]]
    param_names = ['size']
    
    def setup(self, size):
        self.v1 = BitVector([i % 2 == 0 for i in range(size)])
        self.v2 = BitVector([(i + 1) % 2 == 0 for i in range(size)])
        
        if HAS_BITARRAY:
            self.ba1 = bitarray([i % 2 == 0 for i in range(size)])
            self.ba2 = bitarray([(i + 1) % 2 == 0 for i in range(size)])
    
    def time_bitvector_xor(self, size):
        _ = self.v1 ^ self.v2
    
    def time_bitvector_xor_inplace(self, size):
        self.v1 ^= self.v2
    
    def time_bitvector_and(self, size):
        _ = self.v1 & self.v2
    
    def time_bitvector_and_inplace(self, size):
        self.v1 &= self.v2
    
    def time_bitarray_xor(self, size):
        if not HAS_BITARRAY:
            raise NotImplementedError("bitarray not installed")
        _ = self.ba1 ^ self.ba2
    
    def time_bitarray_xor_inplace(self, size):
        if not HAS_BITARRAY:
            raise NotImplementedError("bitarray not installed")
        self.ba1 ^= self.ba2
    
    def time_bitarray_and(self, size):
        if not HAS_BITARRAY:
            raise NotImplementedError("bitarray not installed")
        _ = self.ba1 & self.ba2
    
    def time_bitarray_and_inplace(self, size):
        if not HAS_BITARRAY:
            raise NotImplementedError("bitarray not installed")
        self.ba1 &= self.ba2


class BitVectorIndexingAndSlicing:
    params = [[100, 1000, 10000]]
    param_names = ['size']
    
    def setup(self, size):
        self.v = BitVector.zeros(size)
        if HAS_BITARRAY:
            self.ba = bitarray([False] * size)
    
    def time_bitvector_getitem(self, size):
        _ = self.v[size // 2]
    
    def time_bitvector_slice(self, size):
        _ = self.v[10:size-10]
    
    def time_bitvector_slice_step(self, size):
        _ = self.v[::2]
    
    def time_bitarray_getitem(self, size):
        if not HAS_BITARRAY:
            raise NotImplementedError("bitarray not installed")
        _ = self.ba[size // 2]
    
    def time_bitarray_slice(self, size):
        if not HAS_BITARRAY:
            raise NotImplementedError("bitarray not installed")
        _ = self.ba[10:size-10]
    
    def time_bitarray_slice_step(self, size):
        if not HAS_BITARRAY:
            raise NotImplementedError("bitarray not installed")
        _ = self.ba[::2]


class BitVectorAggregations:
    params = [[100, 1000, 10000]]
    param_names = ['size']
    
    def setup(self, size):
        self.v1 = BitVector([i % 2 == 0 for i in range(size)])
        self.v2 = BitVector([(i + 1) % 2 == 0 for i in range(size)])
        if HAS_BITARRAY:
            self.ba = bitarray([False] * size)
    
    def time_bitvector_weight(self, size):
        _ = self.v1.weight
    
    def time_bitvector_dot(self, size):
        _ = self.v1.dot(self.v2)
    
    def time_bitarray_count(self, size):
        if not HAS_BITARRAY:
            raise NotImplementedError("bitarray not installed")
        _ = self.ba.count()


class BitVectorIteration:
    params = [[100, 1000, 10000]]
    param_names = ['size']
    
    def setup(self, size):
        self.v = BitVector.zeros(size)
        if HAS_BITARRAY:
            self.ba = bitarray([False] * size)
    
    def time_bitvector_iter(self, size):
        for bit in self.v:
            pass
    
    def time_bitarray_iter(self, size):
        if not HAS_BITARRAY:
            raise NotImplementedError("bitarray not installed")
        for bit in self.ba:
            pass
