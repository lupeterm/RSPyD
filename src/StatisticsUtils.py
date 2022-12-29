import numpy as np
from typing import Tuple

class StatisticSUtils:
    def __init__(self, buffer_size: int) -> None:
        self.databuffer = np.zeros(buffer_size)
        self.tmp_buffer = np.zeros(buffer_size)

    def size(self, size: int):
        self._size = size
        if len(self.databuffer) < size:
            self.databuffer = np.pad(self.databuffer, (0,size-len(self.databuffer),'constant'))
            self.tmp_buffer = np.pad(self.tmp_buffer, (0,size-len(self.tmp_buffer),'constant'))

    def get_median(self):
        self.tmp_buffer[:self._size] = self.databuffer[:self._size]
        return np.partition(self.tmp_buffer[:self._size], kth=self._size//2)[self._size//2]

    def get_min_max_R(self, range)->Tuple[float,float]:
        median = self.get_median()
        mad = self.get_MAD(median)
        return median - range*mad, median + range*mad

    def get_MAD(self, median):
        for i in range(self._size):
            self.tmp_buffer[i] = abs(self.databuffer[i]- median)
        return 1.4826 * np.partition(self.tmp_buffer[:self._size], self._size//2)[self._size//2]