import numpy as np
from typing import Dict, ItemsView, KeysView, Union, ValuesView

Index = Union[int, slice, list, np.ndarray]


def _normalize(index: Index) -> Index:
    """
    Turns a bare int into a length-1 slice so that indexing along an axis
    never drops that axis (arrays stay 2D: (horizon, width)).
    """
    if isinstance(index, (int, np.integer)):
        return slice(index, index + 1 if index != -1 else None)
    return index


class _AxisIndexer:
    """
    Indexer bound to one axis (time or bus) of a Trajectory. Slicing always
    returns a Trajectory; every key is sliced the same way.
    """

    def __init__(self, parent: "Trajectory", axis: int):
        self._parent = parent
        self._axis = axis

    def __getitem__(self, index: Index) -> "Trajectory":
        index = _normalize(index)
        if self._axis == 0:
            data = {k: v[index, :] for k, v in self._parent.data.items()}
        else:
            data = {k: v[:, index] for k, v in self._parent.data.items()}
        return Trajectory(data)

    def __setitem__(self, index: Index, value: "Trajectory") -> None:
        index = _normalize(index)
        self._parent._check_compatible(value, axis=self._axis)
        for k, v in self._parent.data.items():
            if self._axis == 0:
                v[index, :] = value.data[k]
            else:
                v[:, index] = value.data[k]


class Trajectory:
    """
    A dict-like container mapping str -> np.ndarray of shape
    (horizon, width), representing a finite-horizon trajectory of
    steady-state quantities (voltage, current, frequency, ...).

    Signal width varies by key: electrical signals (voltage, current, ...)
    are indexed per-bus (width n_buses), while mechanical signals (rotor
    angle, frequency, ...) exist only at generator buses (width n_gens).
    Generator buses are assumed to be the first n_gens bus indices, so a
    bus index/slice that stays within range is valid for both. All keys
    must still share the same time horizon.

    Arrays may be real or complex independently per key (e.g. "v"/"i" as
    complex phasors, "freq"/"p_mech" as real signals) - dtype is tracked
    per-key by numpy, nothing extra is needed to support the mix.

    Three ways to access the underlying data:
      - traj["v"]        -> raw (horizon, width) array for that key
      - traj.at_time[t]   -> Trajectory sliced to time index/slice t (all buses)
      - traj.at_bus[i]    -> Trajectory sliced to bus index/slice i (full horizon)
    """

    def __init__(self, data: Dict[str, np.ndarray]):
        self.data = {k: np.asarray(v) for k, v in data.items()}

        for k, v in self.data.items():
            if v.ndim != 2:
                raise ValueError(f"Trajectory arrays must be 2D (horizon, width); key {k!r} has shape {v.shape}")

        horizons = {v.shape[0] for v in self.data.values()}
        if len(horizons) > 1:
            shapes = {k: v.shape for k, v in self.data.items()}
            raise ValueError(f"All keys must share the same time horizon, got {shapes}")

    def width(self, key: str) -> int:
        return self.data[key].shape[1]

    @property
    def horizon(self) -> int:
        return next(iter(self.data.values())).shape[0]

    @property
    def n_buses(self) -> int:
        return max(v.shape[1] for v in self.data.values())

    @property
    def at_time(self) -> _AxisIndexer:
        return _AxisIndexer(self, axis=0)

    @property
    def at_bus(self) -> _AxisIndexer:
        return _AxisIndexer(self, axis=1)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.data[key]

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        value = np.asarray(value)
        if value.ndim != 2 or value.shape[0] != self.horizon:
            raise ValueError(
                f"Value for key {key!r} must have shape (horizon={self.horizon}, width), got {value.shape}"
            )
        self.data[key] = value

    def keys(self) -> KeysView[str]:
        return self.data.keys()

    def items(self) -> ItemsView[str, np.ndarray]:
        return self.data.items()

    def values(self) -> ValuesView[np.ndarray]:
        return self.data.values()

    def copy(self) -> "Trajectory":
        return Trajectory({k: v.copy() for k, v in self.data.items()})

    def zeroslike(self) -> "Trajectory":
        return Trajectory({k: np.zeros_like(v) for k, v in self.data.items()})

    def _check_compatible(self, other: "Trajectory", axis: "int | None" = None) -> None:
        if self.data.keys() != other.data.keys():
            raise ValueError(f"Trajectories have mismatched keys: {self.data.keys()} vs {other.data.keys()}")
        if axis is None or axis == 1:
            if self.horizon != other.horizon:
                raise ValueError(f"Trajectories have mismatched horizons: {self.horizon} vs {other.horizon}")
        if axis is None or axis == 0:
            mismatched = {
                k: (self.width(k), other.width(k)) for k in self.data if self.width(k) != other.width(k)
            }
            if mismatched:
                raise ValueError(f"Trajectories have mismatched widths for keys: {mismatched}")

    def __add__(self, other: "Trajectory") -> "Trajectory":
        self._check_compatible(other)
        return Trajectory({k: self.data[k] + other.data[k] for k in self.data.keys()})

    def __sub__(self, other: "Trajectory") -> "Trajectory":
        self._check_compatible(other)
        return Trajectory({k: self.data[k] - other.data[k] for k in self.data.keys()})

    def __mul__(self, a: float) -> "Trajectory":
        return Trajectory({k: a * v for k, v in self.data.items()})

    def __rmul__(self, a: float) -> "Trajectory":
        return self.__mul__(a)

    def __neg__(self) -> "Trajectory":
        return self.__mul__(-1)

    def norm(self) -> float:
        return float(np.linalg.norm(np.concatenate([v.ravel() for v in self.data.values()])))
