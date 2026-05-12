from typing import Callable, Any

import numpy as np

class LazyList:
    def __init__(self, generator_func: Callable[[int], Any], length: int):
        self.generator_func = generator_func
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.generator_func(i) for i in range(*index.indices(self.length))]
        elif isinstance(index, (int, np.integer)):
            index = int(index)
            if index < 0:
                index += self.length
            if index < 0 or index >= self.length:
                raise IndexError("LazyList index out of range")
            return self.generator_func(index)
        else:
            raise TypeError("LazyList indices must be integers or slices")

    def __iter__(self):
        for i in range(self.length):
            yield self.generator_func(i)
