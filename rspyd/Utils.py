"""
Included in this file are mostly basic geometric functions.
Why not simply use numpy? Because I benchmarked it (using cProfile):

rspyd_dot:
ncalls  tottime  percall  cumtime  percall function
125685    0.081    0.000    0.081    0.000 rspyd_dot

np.dot:
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
125685    0.113    0.000    0.223    0.000 <__array_function__ internals>:177(dot)

Lastly, the decorator used for these benchmarks is included at the end of the file.
"""
from typing import Callable, Tuple
import cProfile, pstats
import io
import numpy as np

####Geometry Functions
def rspyd_dot(a,b)->float:
    """
    a, b: np.ndarray[3,1] or List[float] with len=3
    """
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def rspyd_cross(a, b) -> np.ndarray:
    """
    a, b: np.ndarray[3,1] or List[float] with len = 3
    """
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ]) 

def rspyd_norm(a) -> np.ndarray:
    return a / rspyd_eucl(a)

def rspyd_eucl(a):
    return np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)

def rspyd_orthogonal_base(n) -> Tuple[np.ndarray, np.ndarray]:
    """
    n: np.ndarray[3,1] or List[float] with len = 3
    """
    base_u = [n[1]-n[2], -n[0], n[0]]
    base_u = rspyd_norm(base_u)
    base_v = rspyd_cross(n, base_u)
    base_v = rspyd_norm(base_v)
    return base_u, base_v

#### Benchmarking Decorator
def bench(func_to_profile: Callable):
    """
    Wraps cProfile benchmarking around given input Callable func_to_profile.
    Returns whatever func_to_profile would return.
    """
    def wrapper(self,*args):
        pr = cProfile.Profile()
        pr.enable()
        ret_val = func_to_profile(self,*args)
        pr.disable()
        s = io.StringIO()
        sortby = pstats.SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return ret_val
    return wrapper