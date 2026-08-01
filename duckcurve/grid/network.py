"""IEEE 33-bus network model.

Primary path: build a pandapower network from the Baran-Wu parameters in
duckcurve.data.ieee33 and run AC power flow. Fallback: a self-contained
backward/forward sweep (BFS) load-flow for the radial topology, so the code
still runs on machines without pandapower.

Note: the duck-curve objective is a power-balance quantity and does not depend on
voltage/loss accuracy, so the BFS fallback is sufficient for Figure 4. The
SALEDI surrogate uses BFS-level outage analysis (downstream-of-cut load).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from ..data.ieee33 import IEEE33_BUSES, IEEE33_LINES, Bus, Line

try:                                       # pragma: no cover - optional dep
    import pandapower as pp                # noqa: F401
    HAS_PANDAPOWER = True
except Exception:                          # pragma: no cover
    HAS_PANDAPOWER = False


@dataclass
class PFResult:
    """Power-flow result for one timestep."""
    bus_voltages_pu: np.ndarray   # shape (33,)
    losses_kw: float
    total_load_kw: float
    total_gen_kw: float           # net slack injection (negative = export)


def build_network(base_kv: float = 12.66, base_mva: float = 10.0):
    """Return a pandapower network if available, else a dict describing the topology.

    The returned object is consumed by :func:`run_power_flow`, which knows how to
    dispatch on type.
    """
    if HAS_PANDAPOWER:
        import pandapower as pp
        net = pp.create_empty_network(sn_mva=base_mva)
        bus_objs: Dict[int, int] = {}
        for b in IEEE33_BUSES:
            bus_objs[b.idx] = pp.create_bus(net, vn_kv=base_kv, name=f"Bus {b.idx}")
        pp.create_ext_grid(net, bus=bus_objs[1], vm_pu=1.0, name="Slack")
        for b in IEEE33_BUSES:
            if b.idx == 1 or (b.p_kw == 0 and b.q_kvar == 0):
                continue
            pp.create_load(
                net, bus=bus_objs[b.idx],
                p_mw=b.p_kw / 1000.0, q_mvar=b.q_kvar / 1000.0,
                name=f"Load {b.idx}",
            )
        z_base_ohm = (base_kv ** 2) / base_mva
        for ln in IEEE33_LINES:
            r_pu = ln.r_ohm / z_base_ohm
            x_pu = ln.x_ohm / z_base_ohm
            pp.create_line_from_parameters(
                net,
                from_bus=bus_objs[ln.from_bus], to_bus=bus_objs[ln.to_bus],
                length_km=1.0,
                r_ohm_per_km=ln.r_ohm, x_ohm_per_km=ln.x_ohm,
                c_nf_per_km=0.0, max_i_ka=1.0,
                name=f"L{ln.from_bus}-{ln.to_bus}",
            )
        net._bus_map = bus_objs           # type: ignore[attr-defined]
        return net
    # Fallback: a plain-Python topology descriptor.
    return {
        "buses": IEEE33_BUSES,
        "lines": IEEE33_LINES,
        "base_kv": base_kv,
        "base_mva": base_mva,
    }


# --------------------------------------------------------------------------
# Backward/forward sweep load flow (radial only).
# --------------------------------------------------------------------------

def _build_adjacency(lines: List[Line]) -> Tuple[Dict[int, List[int]], Dict[int, Line]]:
    """Return (parent_to_children, line_to_child_bus). Slack is the root."""
    children: Dict[int, List[int]] = {}
    line_by_to_bus: Dict[int, Line] = {}
    for ln in lines:
        children.setdefault(ln.from_bus, []).append(ln.to_bus)
        line_by_to_bus[ln.to_bus] = ln
    return children, line_by_to_bus


def _bfs_topo_order(children: Dict[int, List[int]], root: int = 1) -> List[int]:
    order: List[int] = [root]
    queue = [root]
    while queue:
        node = queue.pop(0)
        for c in children.get(node, []):
            order.append(c)
            queue.append(c)
    return order


def _bfs_power_flow(
    topology: dict,
    bus_p_kw: Dict[int, float],
    bus_q_kvar: Dict[int, float],
    max_iter: int = 30,
    tol: float = 1e-5,
) -> PFResult:
    """Distflow-style backward/forward sweep for a radial feeder.

    Active and reactive nodal injections at each bus are passed in. Positive = load,
    negative = generation. Slack bus injection comes out as `total_gen_kw`.
    """
    lines = topology["lines"]
    base_kv = topology["base_kv"]
    children, line_by_to = _build_adjacency(lines)
    order = _bfs_topo_order(children, root=1)

    # Voltages in kV, complex. Initial guess = nominal at every bus.
    v_kv: Dict[int, complex] = {b.idx: complex(base_kv, 0.0) for b in topology["buses"]}

    for _it in range(max_iter):
        # Backward sweep: accumulate downstream complex power into each line.
        s_line: Dict[int, complex] = {}
        for bus in reversed(order):
            s_bus = complex(bus_p_kw.get(bus, 0.0), bus_q_kvar.get(bus, 0.0))
            s = s_bus
            for c in children.get(bus, []):
                s = s + s_line[c]
            if bus != 1:                              # the line feeding this bus
                ln = line_by_to[bus]
                z = complex(ln.r_ohm, ln.x_ohm)
                v_to = v_kv[bus]
                # Line losses S_loss = |I|^2 * Z = |S/V|^2 * Z
                i_mag2 = abs(s) ** 2 / max(abs(v_to) ** 2, 1e-6)
                s_line[bus] = s + complex(i_mag2 * z.real, i_mag2 * z.imag)
            else:
                s_line[bus] = s

        # Forward sweep: update voltages.
        v_kv_new: Dict[int, complex] = {1: complex(base_kv, 0.0)}
        for bus in order[1:]:
            ln = line_by_to[bus]
            v_from = v_kv_new[ln.from_bus]
            s_down = s_line[bus]                      # kVA flowing into "to" bus
            z = complex(ln.r_ohm, ln.x_ohm) / 1000.0  # ohms → kV²/kVA scaling
            # ΔV ≈ (P·R + Q·X) / V  for distribution feeders
            dv = (s_down.real * ln.r_ohm + s_down.imag * ln.x_ohm) / (abs(v_from) * 1000.0)
            v_to = v_from - dv
            v_kv_new[bus] = complex(v_to, 0.0)

        # Convergence check.
        max_dv = max(abs(v_kv_new[b] - v_kv[b]) for b in v_kv)
        v_kv = v_kv_new
        if max_dv < tol:
            break

    # Assemble result.
    vm_pu = np.array([abs(v_kv[b.idx]) / base_kv for b in topology["buses"]])
    total_load = sum(bus_p_kw.values())
    losses = float(s_line[1].real - total_load)
    return PFResult(
        bus_voltages_pu=vm_pu,
        losses_kw=losses,
        total_load_kw=total_load,
        total_gen_kw=float(s_line[1].real),
    )


def run_power_flow(
    network,
    bus_p_kw: Dict[int, float],
    bus_q_kvar: Dict[int, float],
) -> PFResult:
    """Run power flow with the given bus injections (positive = load).

    Dispatches on `network` type — uses pandapower if it was used to build the
    network, BFS otherwise.
    """
    if HAS_PANDAPOWER and not isinstance(network, dict):
        import pandapower as pp
        bus_map = network._bus_map                    # type: ignore[attr-defined]
        # Reset and write loads.
        network.load.drop(network.load.index, inplace=True)
        for b_idx, p in bus_p_kw.items():
            q = bus_q_kvar.get(b_idx, 0.0)
            if p == 0 and q == 0:
                continue
            pp.create_load(
                network, bus=bus_map[b_idx],
                p_mw=p / 1000.0, q_mvar=q / 1000.0,
            )
        pp.runpp(network, algorithm="nr", calculate_voltage_angles=False, init="flat")
        vm_pu = network.res_bus["vm_pu"].to_numpy()
        losses_kw = float(network.res_line["pl_mw"].sum() * 1000.0)
        total_load_kw = float(network.res_load["p_mw"].sum() * 1000.0)
        total_gen_kw = float(network.res_ext_grid["p_mw"].sum() * 1000.0)
        return PFResult(
            bus_voltages_pu=vm_pu,
            losses_kw=losses_kw,
            total_load_kw=total_load_kw,
            total_gen_kw=total_gen_kw,
        )
    # BFS fallback
    return _bfs_power_flow(network, bus_p_kw, bus_q_kvar)
