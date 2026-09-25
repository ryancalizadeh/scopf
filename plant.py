"""
Plant for closed-loop (receding-horizon) simulation of the dynamic OPF.

The controller (mpc.run_closed_loop) hands the plant one step of bus
injections, the first step of its plan; the plant applies them, advances its
own device states, balances the network and reports what was realized. The
plant keeps its own models, so it can differ from the one the controller
optimizes over.

Device models are one class per device type, vectorised over the devices of
that type, with the interface

    step(state, p_cmd, k) -> (next_state, p_realized)

where k is the absolute step index, p_cmd is in device convention (battery
p > 0 discharges, thermal p > 0 heats) and p_realized is the power the devices
actually exchanged with their buses over the step. The linear models below are
exactly the MPC's model; a nonlinear model replaces one of them and may
sub-step internally or realize a power other than the command (losses,
saturation). Any resulting mismatch is balanced by the generators.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from Config import Config


class LinearBattery:
    """q_t = q_{t-1} - dt p_t: the battery model of the MPC (discharge positive)."""

    def __init__(self, config: Config):
        self.dt = config.dt

    def step(self, soc: np.ndarray, p: np.ndarray, k: int):
        return soc - self.dt * p, p.copy()


class LinearThermal:
    """
    T_t = (1 - mu/c) T_{t-1} + (eta/c) p_t + (mu/c) Tamb_t: the thermal model of
    the MPC (p > 0 heats). Temperatures are absolute, in degrees C.
    """

    def __init__(self, config: Config):
        self.decay = 1.0 - config.thermal_mu / config.thermal_c
        self.gain = config.thermal_eta / config.thermal_c
        self.ambient = config.thermal_mu / config.thermal_c
        self.T_amb = config.thermal_T_amb_ext

    def step(self, temp: np.ndarray, p: np.ndarray, k: int):
        return self.decay * temp + self.gain * p + self.ambient * self.T_amb[:, k], p.copy()


class Plant:
    """
    The power system under closed-loop control, in the [gens | loads | thermal
    | batteries] bus layout of Config. Carries the measured state the
    controller reads before each step:

        soc         SOC of each battery
        temp        temperature of each thermal unit (degrees C)
        p_gen_prev  generator output realized at the previous step (None before the first)

    Each step, loads draw their demand (from config, the truth data), the
    device models are stepped, and the mismatch between the realized
    injections is shared by the generators in proportion to gen_P_max
    (participation factors). Bus angles then follow from DC power flow with
    bus 0 as the reference. Nothing is clipped: values outside their bounds are
    recorded as they are, so violations show up in the trajectory.
    """

    def __init__(self, config: Config, battery=None, thermal=None):
        if config.n_gens == 0:
            raise ValueError("the plant balances the network with its generators; config has none")
        self.config = config
        self.battery = battery if battery is not None else LinearBattery(config)
        self.thermal = thermal if thermal is not None else LinearThermal(config)

        n_gens, n_loads, n_thermal = config.n_gens, config.n_loads, config.n_thermal
        self._gen = slice(0, n_gens)
        self._load = slice(n_gens, n_gens + n_loads)
        self._thermal = slice(n_gens + n_loads, n_gens + n_loads + n_thermal)
        self._battery = slice(n_gens + n_loads + n_thermal, config.n_buses)

        self.participation = config.gen_P_max / config.gen_P_max.sum()
        # B_dc = -B is the network Laplacian; with bus 0 removed it is SPD for a
        # connected network, and theta_0 = 0 fixes the angle as in centralized.py.
        self._cho = cho_factor(-config.B[1:, 1:])

        self.k = 0
        self.soc = config.battery_q0.astype(float).copy()
        self.temp = config.thermal_T0.astype(float).copy()
        self.p_gen_prev = None

    def step(self, p_cmd: np.ndarray) -> dict:
        """
        Applies one step of commanded bus injections p_cmd (n_buses,), in the
        net-injection sign convention of the solvers (load-like devices
        negative). Returns the realized step: "p" and "theta" (n_buses,), the
        states after the step "soc" (n_battery,) and "temp" (n_thermal,), and
        "imbalance", the injection mismatch the generators absorbed.
        """
        config, k = self.config, self.k
        if k >= config.N + config.n_lookahead:
            raise IndexError(f"plant has data for {config.N + config.n_lookahead} steps; step {k} requested")
        p_cmd = np.asarray(p_cmd, dtype=float).reshape(config.n_buses)

        soc_next, p_batt = self.battery.step(self.soc, p_cmd[self._battery], k)
        temp_next, p_heat = self.thermal.step(self.temp, -p_cmd[self._thermal], k)

        p = np.empty(config.n_buses)
        p[self._gen] = p_cmd[self._gen]
        p[self._load] = -config.load_P_ext[:, k]
        p[self._thermal] = -p_heat
        p[self._battery] = p_batt
        imbalance = float(p.sum())
        p[self._gen] -= self.participation * imbalance

        theta = np.zeros(config.n_buses)
        theta[1:] = cho_solve(self._cho, p[1:])

        self.soc, self.temp = soc_next, temp_next
        self.p_gen_prev = p[self._gen].copy()
        self.k += 1
        return {"p": p, "theta": theta, "soc": soc_next.copy(), "temp": temp_next.copy(), "imbalance": imbalance}
