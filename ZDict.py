import numpy as np
from typing import Dict, Union, overload

class ZDict:
    """
    A dict-like container mapping str -> np.ndarray that supports elementwise
    arithmetic (+, -, *, unary -) and a norm()
    """

    def __init__(self, data: Dict[str, np.ndarray]):
        self.data = dict(data)

    @overload
    def __getitem__(self, key: str) -> np.ndarray: ...
    @overload
    def __getitem__(self, key: Union[int, slice]) -> "ZDict": ...

    def __getitem__(self, key):
        """
        Indexing by str returns the underlying array for that key. Indexing
        by anything else (int, slice, ...) applies the same index to every
        value and returns a ZDict of the results.
        """
        if isinstance(key, str):
            return self.data[key]
        return ZDict({k: v[key] for k, v in self.data.items()})

    @overload
    def __setitem__(self, key: str, value: np.ndarray) -> None: ...
    @overload
    def __setitem__(self, key: Union[int, slice], value: "ZDict") -> None: ...

    def __setitem__(self, key, value) -> None:
        """
        Setting by str replaces the array for that key. Setting by anything
        else assigns element-wise from another ZDict with matching keys.
        """
        if isinstance(key, str):
            self.data[key] = value
        else:
            for k in self.data.keys():
                self.data[k][key] = value[k]

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()

    def copy(self) -> "ZDict":
        return ZDict({k: v.copy() for k, v in self.data.items()})

    def zeroslike(self) -> "ZDict":
        return ZDict({k: np.zeros_like(v) for k, v in self.data.items()})

    def __add__(self, other: "ZDict") -> "ZDict":
        return ZDict({k: self.data[k] + other.data[k] for k in self.data.keys()})

    def __sub__(self, other: "ZDict") -> "ZDict":
        return ZDict({k: self.data[k] - other.data[k] for k in self.data.keys()})

    def __mul__(self, a: float) -> "ZDict":
        return ZDict({k: a * v for k, v in self.data.items()})

    def __rmul__(self, a: float) -> "ZDict":
        return self.__mul__(a)

    def __neg__(self) -> "ZDict":
        return self.__mul__(-1)

    def norm(self) -> float:
        return float(np.linalg.norm(np.concatenate([v.ravel() for v in self.data.values()])))