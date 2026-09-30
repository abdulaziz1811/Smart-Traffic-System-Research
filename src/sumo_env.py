"""
SUMO Closed-Loop Environment
=============================
Same interface as TrafficSignalEnv (31-dim observation, cyclic/free
actions, min/max green, yellow time), but the traffic is simulated by the
SUMO microscopic simulator: individual vehicles with acceleration,
braking, gaps and start-up delays instead of the fluid queue model.

Agents trained in the fast fluid simulator can be evaluated here without
retraining (sim-to-sim transfer), and every signal decision changes what
happens next (closed loop), which a recorded video cannot do.

Network (built once with netconvert): 4 approaches x 2 incoming lanes,
250 m long. Lane 0 of every approach is straight-only, lane 1 is a
protected left turn -- the same 8 lanes and 4 phases as the fluid model:
    0 N straight, 1 N left, 2 S straight, 3 S left,
    4 E straight, 5 E left, 6 W straight, 7 W left

Queues = halting vehicles on each incoming lane plus vehicles waiting to
be inserted (spill-back beyond 250 m). Arrivals: per-second Bernoulli
with the episode's per-lane rates (same rates as the fluid model).

Requires:  pip install eclipse-sumo traci sumolib
"""

import os
import shutil
import subprocess
import tempfile
from typing import Optional

import numpy as np

from src.environment import TrafficSignalEnv

# Incoming lane id and route (edges) for each of the 8 model lanes
LANES = [
    ("N2C_0", "N2C C2S"),  # 0 N straight (southbound)
    ("N2C_1", "N2C C2E"),  # 1 N left
    ("S2C_0", "S2C C2N"),  # 2 S straight (northbound)
    ("S2C_1", "S2C C2W"),  # 3 S left
    ("E2C_0", "E2C C2W"),  # 4 E straight (westbound)
    ("E2C_1", "E2C C2S"),  # 5 E left
    ("W2C_0", "W2C C2E"),  # 6 W straight (eastbound)
    ("W2C_1", "W2C C2N"),  # 7 W left
]

_NODES = """<nodes>
    <node id="C" x="0" y="0" type="traffic_light" tl="C"/>
    <node id="N" x="0" y="{L}"/>
    <node id="S" x="0" y="-{L}"/>
    <node id="E" x="{L}" y="0"/>
    <node id="W" x="-{L}" y="0"/>
</nodes>
"""

_EDGES = """<edges>
""" + "".join(
    f'    <edge id="{a}2C" from="{a}" to="C" numLanes="2" speed="13.89"/>\n'
    f'    <edge id="C2{a}" from="C" to="{a}" numLanes="2" speed="13.89"/>\n'
    for a in "NSEW") + "</edges>\n"

_CONNECTIONS = """<connections>
""" + "".join(
    f'    <connection from="{lane.split("_")[0]}" to="{route.split()[1]}" '
    f'fromLane="{lane.split("_")[1]}" toLane="{lane.split("_")[1]}"/>\n'
    for lane, route in LANES) + "</connections>\n"


def sumo_home() -> str:
    """Locate SUMO (SUMO_HOME or the eclipse-sumo pip package)."""
    if os.environ.get("SUMO_HOME"):
        return os.environ["SUMO_HOME"]
    try:
        import sumo
        os.environ["SUMO_HOME"] = sumo.SUMO_HOME
        return sumo.SUMO_HOME
    except ImportError:
        raise RuntimeError("SUMO not found: pip install eclipse-sumo traci sumolib, or set SUMO_HOME")


def _binary(name):
    path = os.path.join(sumo_home(), "bin", name)
    return path if os.path.exists(path) else (shutil.which(name) or name)


def build_network(out_dir, approach_length=250):
    """Write and netconvert the 4-way intersection. Returns the .net.xml path."""
    os.makedirs(out_dir, exist_ok=True)
    net = os.path.join(out_dir, "intersection.net.xml")
    if os.path.exists(net):
        return net
    files = {"nodes.nod.xml": _NODES.format(L=approach_length),
             "edges.edg.xml": _EDGES, "conn.con.xml": _CONNECTIONS}
    for name, text in files.items():
        with open(os.path.join(out_dir, name), "w") as f:
            f.write(text)
    subprocess.run([
        _binary("netconvert"),
        "--node-files", os.path.join(out_dir, "nodes.nod.xml"),
        "--edge-files", os.path.join(out_dir, "edges.edg.xml"),
        "--connection-files", os.path.join(out_dir, "conn.con.xml"),
        "--no-turnarounds", "true", "--no-warnings", "true",
        "-o", net,
    ], check=True, capture_output=True)
    return net


class SumoTrafficSignalEnv(TrafficSignalEnv):
    """TrafficSignalEnv whose traffic model is a SUMO simulation."""

    def __init__(self, cfg: dict, action_mode: Optional[str] = None, gui: bool = False,
                 work_dir: Optional[str] = None):
        super().__init__(cfg, action_mode)
        import traci  # noqa: F401  (fail early if missing)
        sumo_home()
        self.gui = gui
        self._own_dir = work_dir is None
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="sumo_tsc_")
        self.net_file = build_network(self.work_dir)
        self._label = f"tsc_{id(self)}"
        self._conn = None
        self._lane_ids = [lane for lane, _ in LANES]
        self._link_lane = None           # TLS link index -> model lane (or None)
        self._on_lane = [set() for _ in LANES]
        self._arrived_total = 0

    # ------------------------------------------------------------------
    #  SUMO process management
    # ------------------------------------------------------------------

    def _write_routes(self, path):
        lines = ['<routes>',
                 '    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" '
                 'maxSpeed="13.89" lcSpeedGain="0" lcCooperative="0"/>']
        for i, (lane, route) in enumerate(LANES):
            lines.append(f'    <route id="r{i}" edges="{route}"/>')
        for i, (lane, _) in enumerate(LANES):
            p = float(np.clip(self.arrivals[i], 0.0, 1.0))
            if p > 0:
                lines.append(
                    f'    <flow id="f{i}" type="car" route="r{i}" begin="0" end="{self.max_steps + 1}" '
                    f'probability="{p:.4f}" departLane="{lane.split("_")[1]}" departSpeed="max"/>')
        lines.append('</routes>')
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def _start(self, seed):
        import traci
        self.close()
        routes = os.path.join(self.work_dir, f"routes_{self._label}.rou.xml")
        self._write_routes(routes)
        cmd = [_binary("sumo-gui" if self.gui else "sumo"),
               "-n", self.net_file, "-r", routes,
               "--step-length", "1", "--seed", str(int(seed)),
               "--time-to-teleport", "-1", "--no-step-log", "true", "--no-warnings", "true",
               "--collision.action", "none"]
        traci.start(cmd, label=self._label)
        self._conn = traci.getConnection(self._label)

        # Map every signal link to the model lane it serves
        links = self._conn.trafficlight.getControlledLinks("C")
        self._link_lane = []
        for group in links:
            in_lane = group[0][0] if group else None
            self._link_lane.append(self._lane_ids.index(in_lane) if in_lane in self._lane_ids else None)
        self._on_lane = [set() for _ in LANES]
        self._arrived_total = 0

    def close(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __del__(self):
        self.close()
        if getattr(self, "_own_dir", False):
            shutil.rmtree(getattr(self, "work_dir", ""), ignore_errors=True)

    # ------------------------------------------------------------------
    #  Gym API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        sumo_seed = seed if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
        self._start(sumo_seed)
        self._read_queues()
        return self._obs(), info

    def _signal_state(self, active_lanes):
        in_clearance = self.clearance_left > 0 or not active_lanes
        state = []
        for lane in self._link_lane:
            if lane is not None and lane in active_lanes and lane not in self.blocked_lanes:
                state.append("G")
            elif lane is not None and in_clearance and lane in self.yellow_lanes:
                state.append("y")
            else:
                state.append("r")
        return "".join(state)

    def _read_queues(self):
        c = self._conn
        q = np.array([c.lane.getLastStepHaltingNumber(l) for l in self._lane_ids], dtype=np.float32)
        # Vehicles that could not enter because the lane is full (spill-back)
        for vid in c.simulation.getPendingVehicles():
            lane = int(vid.split(".")[0][1:]) if vid.startswith("f") else None
            if lane is not None and 0 <= lane < len(q):
                q[lane] += 1
        self.queues = q

    def _simulate_traffic(self, active_lanes):
        c = self._conn
        cars_in_green = float(sum(self.queues[l] for l in active_lanes))
        c.trafficlight.setRedYellowGreenState("C", self._signal_state(active_lanes))
        c.simulationStep()

        # Served = vehicles that left an incoming lane (crossed the stop line)
        served = 0
        for i, lane in enumerate(self._lane_ids):
            now = set(c.lane.getLastStepVehicleIDs(lane))
            served += len(self._on_lane[i] - now)
            self._on_lane[i] = now
        self._arrived_total += c.simulation.getArrivedNumber()

        self._read_queues()
        self.waits = np.where(np.isin(np.arange(self.n_app), active_lanes),
                              self.waits * 0.95, self.waits) + self.queues
        return cars_in_green, float(served)

    def _info(self):
        info = super()._info()
        info["arrived"] = self._arrived_total
        return info
