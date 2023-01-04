import numpy as np


class UnionFind:
    def __init__(self, size: int) -> None:
        self.groups = np.full(size, -1)

    def root(self, x: int) -> int:
        r = x
        while self.groups[r] >= 0:
            r = self.groups[r]
        while self.groups[x] >= 0:
            tmp = self.groups[x]
            self.groups[x] = r
            x = tmp
        return r


    def join(self, x: int, y: int) -> None:
        x = self.root(x)
        y = self.root(y)

        if x != y:
            if self.groups[x] < self.groups[y]:
                self.groups[x] += self.groups[y]
                self.groups[y] = x
            else:
                self.groups[y] += self.groups[x]
                self.groups[x] = y

    def connected(self, x: int, y: int) -> bool:
        return self.root(x) == self.root(y)
