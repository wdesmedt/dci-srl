# © 2025 Nokia
# Licensed under the BSD 3-Clause License
# SPDX-License-Identifier: BSD-3-Clause
"""
Shared fixtures / helpers for the DCI validated-design test suite.

The tests drive the *running* containerlab topology over `docker exec`, so the
lab must already be deployed (`containerlab deploy -t dci-without-eda.clab.yaml`)
on the same host where pytest runs.

Nothing here is SR-Linux-release specific beyond the `sr_cli` CLI and the
client bootstrap (netns + 802.1q + LACP bonds) created by base-configs/*.sh.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

# Set DCI_TRACE=1 to echo every shell command the suite runs (and any piped
# sr_cli config payload). Combine with `pytest -s` so the trace reaches the
# terminal, e.g.  DCI_TRACE=1 pytest -v -s -m convergence
_TRACE = os.environ.get("DCI_TRACE")

import pytest

# containerlab topology that defines this lab (resolved so fcli works from any cwd)
CLAB_TOPO = Path(__file__).resolve().parent.parent / "dci-without-eda.clab.yaml"

# --------------------------------------------------------------------------- #
# Topology facts (kept in one place so tests stay declarative)
# --------------------------------------------------------------------------- #

DC1_DCGWS = ["dcgw1", "dcgw2"]
DC2_DCGWS = ["dcgw3", "dcgw4"]
ALL_DCGWS = DC1_DCGWS + DC2_DCGWS

# the L3-DCI tenant IP-VRF (one shared name on every leaf + DCGW)
L3DCI_NI = "ipvrf-l3dci"

# leaf VTEPs per DC + the underlay subnet each DC's VTEPs live in. Used to assert
# that remote-DC destinations are only ever reached via the LOCAL DCGW pair
# (i.e. no raw VXLAN route leaks end-to-end across the DCI).
DC1_LEAVES = ["leaf1", "leaf2", "leaf3", "leaf4"]
DC2_LEAVES = ["leaf5", "leaf6", "leaf7", "leaf8"]
DC_LEAVES = {1: DC1_LEAVES, 2: DC2_LEAVES}
DC_DCGWS = {1: DC1_DCGWS, 2: DC2_DCGWS}
DC_VTEP_SUBNET = {1: "192.0.2.", 2: "192.0.3."}
LOCAL_DCGW_VTEPS = {1: {"192.0.2.151", "192.0.2.152"},
                    2: {"192.0.3.153", "192.0.3.154"}}

# direct DCGW<->DCGW mesh ports (IS-IS metric 10) per node
DIRECT_MESH_PORTS = {
    "dcgw1": ["ethernet-1/3", "ethernet-1/4", "ethernet-1/5"],
    "dcgw2": ["ethernet-1/3", "ethernet-1/4", "ethernet-1/5"],
    "dcgw3": ["ethernet-1/3", "ethernet-1/4", "ethernet-1/5"],
    "dcgw4": ["ethernet-1/3", "ethernet-1/4", "ethernet-1/5"],
}
# every fabric/DCI-facing port on a DCGW (e1-1..e1-7) - used to isolate a GW
DCGW_ALL_PORTS = [f"ethernet-1/{i}" for i in range(1, 8)]

# WAN / DCI MPLS transports that run in parallel over the same IS-IS core. Both an
# LDP and an SR-ISIS tunnel exist to every remote loopback; LDP is preferred
# (tunnel-table preference 9 < SR-ISIS 11), so disabling one shifts to the other.
WAN_TRANSPORTS = ("ldp", "sr-isis")
P_ROUTERS = ["p1", "p2"]
WAN_NODES = ALL_DCGWS + P_ROUTERS          # DCGWs terminate services; p1/p2 are LSRs
# system0 loopback (BGP/transport endpoint) per DCGW
DCGW_SYS0 = {"dcgw1": "192.0.2.151", "dcgw2": "192.0.2.152",
             "dcgw3": "192.0.3.153", "dcgw4": "192.0.3.154"}
# IS-IS SR node-SID index per WAN node (== last octet of system0, so the SR label
# is SRGB-base 15000 + index). Needed to re-apply SR config in teardown.
WAN_NODE_SID = {"dcgw1": 151, "dcgw2": 152, "dcgw3": 153, "dcgw4": 154,
                "p1": 201, "p2": 202}
# the two "diagonals" of the 2x2 gateway grid (DC1={dcgw1,dcgw2}, DC2={dcgw3,dcgw4}):
# each pair fails exactly ONE gateway per DC, and (given BD-A->dcgw1/3, BD-B->dcgw2/4)
# forces a primary-service failover in BOTH DCs at once.
DIAGONAL_DCGW_PAIRS = [("dcgw1", "dcgw4"), ("dcgw2", "dcgw3")]


def remote_dcgw_loopbacks(node: str) -> set[str]:
    """system0 loopbacks of the DCGWs in the OTHER DC (the DCI tunnel endpoints)."""
    peers = DC2_DCGWS if node in DC1_DCGWS else DC1_DCGWS
    return {DCGW_SYS0[p] for p in peers}


@dataclass(frozen=True)
class Endpoint:
    """A logical client (netns) inside a network-multitool container."""

    container: str
    netns: str
    ip: str
    dc: int
    service: str  # "l2" | "l3"
    homing: str  # "mh" | "sh"


# logical clients created by base-configs/*.sh
ENDPOINTS = {
    # DC1 - ES mh-dc1 (leaf1/2)
    "mh1-l2a": Endpoint("mh-dc1", "mh1-l2a", "10.100.0.11", 1, "l2", "mh"),
    "mh1-l2b": Endpoint("mh-dc1", "mh1-l2b", "10.100.0.12", 1, "l2", "mh"),
    "mh1-l3a": Endpoint("mh-dc1", "mh1-l3a", "10.200.1.11", 1, "l3", "mh"),
    "mh1-l3b": Endpoint("mh-dc1", "mh1-l3b", "10.200.1.12", 1, "l3", "mh"),
    "sh1-l2": Endpoint("sh-dc1", "sh1-l2", "10.100.0.13", 1, "l2", "sh"),
    "sh1-l3": Endpoint("sh-dc1", "sh1-l3", "10.200.1.13", 1, "l3", "sh"),
    # DC1 - L2 DCI BD-B (2nd stretched BD, vlan 110, pinned to dcgw2)
    "mh1-l2c": Endpoint("mh-dc1", "mh1-l2c", "10.110.0.11", 1, "l2b", "mh"),
    "mh1-l2d": Endpoint("mh-dc1", "mh1-l2d", "10.110.0.12", 1, "l2b", "mh"),
    "sh1-l2b": Endpoint("sh-dc1", "sh1-l2b", "10.110.0.13", 1, "l2b", "sh"),
    # DC1 - ES mh-dc1b (leaf3/4)
    "mh1b-l2a": Endpoint("mh-dc1b", "mh1b-l2a", "10.100.0.14", 1, "l2", "mh"),
    "mh1b-l2b": Endpoint("mh-dc1b", "mh1b-l2b", "10.100.0.15", 1, "l2", "mh"),
    "mh1b-l3a": Endpoint("mh-dc1b", "mh1b-l3a", "10.200.1.14", 1, "l3", "mh"),
    "mh1b-l3b": Endpoint("mh-dc1b", "mh1b-l3b", "10.200.1.15", 1, "l3", "mh"),
    # DC2 - ES mh-dc2 (leaf5/6)
    "mh2-l2a": Endpoint("mh-dc2", "mh2-l2a", "10.100.0.21", 2, "l2", "mh"),
    "mh2-l2b": Endpoint("mh-dc2", "mh2-l2b", "10.100.0.22", 2, "l2", "mh"),
    "mh2-l3a": Endpoint("mh-dc2", "mh2-l3a", "10.200.2.21", 2, "l3", "mh"),
    "mh2-l3b": Endpoint("mh-dc2", "mh2-l3b", "10.200.2.22", 2, "l3", "mh"),
    "sh2-l2": Endpoint("sh-dc2", "sh2-l2", "10.100.0.23", 2, "l2", "sh"),
    "sh2-l3": Endpoint("sh-dc2", "sh2-l3", "10.200.2.23", 2, "l3", "sh"),
    # DC2 - L2 DCI BD-B (2nd stretched BD, vlan 110, pinned to dcgw4)
    "mh2-l2c": Endpoint("mh-dc2", "mh2-l2c", "10.110.0.21", 2, "l2b", "mh"),
    "mh2-l2d": Endpoint("mh-dc2", "mh2-l2d", "10.110.0.22", 2, "l2b", "mh"),
    "sh2-l2b": Endpoint("sh-dc2", "sh2-l2b", "10.110.0.23", 2, "l2b", "sh"),
    # DC2 - ES mh-dc2b (leaf7/8)
    "mh2b-l2a": Endpoint("mh-dc2b", "mh2b-l2a", "10.100.0.24", 2, "l2", "mh"),
    "mh2b-l2b": Endpoint("mh-dc2b", "mh2b-l2b", "10.100.0.25", 2, "l2", "mh"),
    "mh2b-l3a": Endpoint("mh-dc2b", "mh2b-l3a", "10.200.2.24", 2, "l3", "mh"),
    "mh2b-l3b": Endpoint("mh-dc2b", "mh2b-l3b", "10.200.2.25", 2, "l3", "mh"),
}

# leaves each access segment (Endpoint.container) attaches to. The host /32 is
# LOCAL on these leaves (connected anycast /24 + ARP), so they do NOT install a
# bgp-evpn-ifl host route; every OTHER node in the DC does. For a multi-homed ES
# the /32 aliases across BOTH listed VTEPs (advertise-ifl-host-ad-routes).
ES_LEAVES = {
    "mh-dc1": ["leaf1", "leaf2"],
    "mh-dc1b": ["leaf3", "leaf4"],
    "sh-dc1": ["leaf3"],
    "mh-dc2": ["leaf5", "leaf6"],
    "mh-dc2b": ["leaf7", "leaf8"],
    "sh-dc2": ["leaf7"],
}

# DCI-facing ports on a DCGW (direct mesh e1-3..5 + WAN uplinks e1-6/7); summed to
# observe how much DCI traffic each gateway forwards.
DCGW_DCI_PORTS = [f"ethernet-1/{i}" for i in range(3, 8)]

# a canary flow used by teardown to wait for the fabric to become healthy again
CANARY_SRC = ENDPOINTS["mh1-l2a"]
CANARY_DST = ENDPOINTS["mh2-l2a"]


# --------------------------------------------------------------------------- #
# low-level docker / sr_cli helpers
# --------------------------------------------------------------------------- #

_name_cache: list[str] | None = None


def _docker_names() -> list[str]:
    global _name_cache
    if _name_cache is None:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=30,
        ).stdout
        _name_cache = out.split()
    return _name_cache


def cname(node: str) -> str:
    """Resolve a topology node to its actual container name (prefix-agnostic)."""
    for n in _docker_names():
        if n == node or n.endswith(f"-{node}"):
            return n
    return node


def _run(cmd: list[str], input_: str | None = None, timeout: int = 90):
    if _TRACE:
        print(f"\n  $ {' '.join(cmd)}", flush=True)
        if input_:
            for ln in input_.splitlines():
                print(f"      | {ln}", flush=True)
    return subprocess.run(
        cmd, input=input_, capture_output=True, text=True, timeout=timeout
    )


def srl(node: str, command: str) -> str:
    """Run a single read-only sr_cli command, return combined output."""
    r = _run(["docker", "exec", cname(node), "sr_cli", command])
    return (r.stdout or "") + (r.stderr or "")


def srl_configure(node: str, lines: list[str]) -> str:
    """Apply config `lines` on `node` (enter candidate / commit now)."""
    payload = "enter candidate\ndiscard stay\n" + "\n".join(lines) + "\ncommit now\n"
    r = _run(["docker", "exec", "-i", cname(node), "sr_cli"], input_=payload)
    out = (r.stdout or "") + (r.stderr or "")
    assert "committed" in out.lower() or "leaving candidate" in out.lower(), (
        f"config commit failed on {node}:\n{out}"
    )
    return out


def set_ports(node: str, ports: list[str], state: str):
    """admin enable/disable a set of interfaces on a node."""
    assert state in ("enable", "disable")
    lines = [f"set / interface {p} admin-state {state}" for p in ports]
    srl_configure(node, lines)


# --------------------------------------------------------------------------- #
# WAN MPLS transport (LDP / SR-ISIS) helpers
# --------------------------------------------------------------------------- #

def dci_tunnels(node: str) -> dict[str, dict[str, str]]:
    """Parse the default IPv4 tunnel-table on `node`.

    Returns {loopback-prefix: {transport: egress-port}} for the ldp/sr-isis
    tunnels (the bordered CLI table is split on the column separator).
    """
    res: dict[str, dict[str, str]] = {}
    out = srl(node, "show network-instance default tunnel-table all")
    for line in out.splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 4:
            continue
        prefix, ttype = cells[1], cells[2]
        if not prefix.endswith("/32") or ttype not in WAN_TRANSPORTS:
            continue
        port = cells[-1] or cells[-2]          # trailing '|' yields an empty last cell
        res.setdefault(prefix, {})[ttype] = port
    return res


# SR-ISIS underlay config lines (per node; the node-SID index is substituted).
def _sr_isis_enable_lines(node: str) -> list[str]:
    idx = WAN_NODE_SID[node]
    return [
        "set / network-instance default segment-routing mpls global-block label-range sr-global",
        "set / network-instance default protocols isis instance main segment-routing mpls static-label-block sr-adj",
        "set / network-instance default protocols isis instance main interface system0.0 "
        f"segment-routing mpls ipv4-node-sid index {idx}",
    ]


# all three SR lines must be removed in ONE commit: the per-node ipv4-node-sid
# depends on the SRGB global-block, so they cannot be deleted piecemeal.
_SR_ISIS_DISABLE_LINES = [
    "delete / network-instance default protocols isis instance main interface system0.0 segment-routing",
    "delete / network-instance default protocols isis instance main segment-routing",
    "delete / network-instance default segment-routing",
]


def disable_wan_transport(transport: str):
    """Disable one MPLS transport (`ldp` | `sr-isis`) on every WAN node."""
    assert transport in WAN_TRANSPORTS
    for node in WAN_NODES:
        if transport == "ldp":
            srl_configure(node, ["set / network-instance default protocols ldp admin-state disable"])
        else:
            srl_configure(node, _SR_ISIS_DISABLE_LINES)


def enable_wan_transport(transport: str):
    """Restore one MPLS transport (`ldp` | `sr-isis`) on every WAN node."""
    assert transport in WAN_TRANSPORTS
    for node in WAN_NODES:
        if transport == "ldp":
            srl_configure(node, ["set / network-instance default protocols ldp admin-state enable"])
        else:
            srl_configure(node, _sr_isis_enable_lines(node))


# --------------------------------------------------------------------------- #
# fcli (fabric CLI) helpers - one gNMI query returns structured state for the
# WHOLE fabric, which is far cleaner than docker-exec + regex for control-plane
# / RIB assertions. Data-plane checks (ping/iperf in client netns, per-port
# counters) still use docker exec because fcli has no notion of them.
# --------------------------------------------------------------------------- #

def fcli_json(report_args: list[str], timeout: int = 90) -> list[dict]:
    """Run an fcli report fabric-wide and return parsed JSON rows. Skips the
    calling test gracefully if fcli is unavailable or the query fails."""
    if shutil.which("fcli") is None:
        pytest.skip("fcli not installed - skipping fabric-wide fcli check")
    cmd = ["fcli", "-t", str(CLAB_TOPO), "-o", "json", *report_args]
    r = _run(cmd, timeout=timeout)
    if r.returncode != 0 or not (r.stdout or "").strip():
        pytest.skip(f"fcli '{' '.join(report_args)}' failed: "
                    f"{(r.stderr or r.stdout or '').strip()[:200]}")
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        pytest.skip(f"fcli '{' '.join(report_args)}' returned non-JSON output")


def l3dci_ipv4_rib() -> list[dict]:
    """Fabric-wide IPv4 RIB rows for the L3-DCI IP-VRF (`fcli ipv4-rib`).

    Each row: {Node, NI, Prefix, next-hop[], type, Act(yes|no), metric, pref, itf[]}.
    Includes non-best paths (Act=no), so a host-route that leaked across the DCI
    cannot hide behind a better path.
    """
    return [r for r in fcli_json(["ipv4-rib"]) if r.get("NI") == L3DCI_NI]


# --------------------------------------------------------------------------- #
# ping helpers + convergence measurement
# --------------------------------------------------------------------------- #

@dataclass
class PingResult:
    transmitted: int
    received: int
    loss_pct: float
    outage_s: float | None = None

    @property
    def ok(self) -> bool:
        return self.received > 0 and self.loss_pct < 100.0


def _parse_stats(text: str) -> PingResult:
    m = re.search(
        r"(\d+) packets transmitted, (\d+)[^\n]*received.*?"
        r"(\d+(?:\.\d+)?)% packet loss",
        text, re.S,
    )
    if not m:
        return PingResult(0, 0, 100.0)
    return PingResult(int(m.group(1)), int(m.group(2)), float(m.group(3)))


def ping(ep_src: Endpoint, dst_ip: str, count: int = 4,
         interval: float = 0.3, wait: int = 2) -> PingResult:
    """Ping `dst_ip` from a client netns; returns parsed stats."""
    cmd = [
        "docker", "exec", cname(ep_src.container),
        "ip", "netns", "exec", ep_src.netns,
        "ping", "-c", str(count), "-i", str(interval), "-W", str(wait), dst_ip,
    ]
    timeout = int(count * interval + wait + 20)
    r = _run(cmd, timeout=timeout)
    return _parse_stats(r.stdout + r.stderr)


def measure_convergence(ep_src: Endpoint, dst_ip: str, action,
                        interval: float = 0.2, pre: float = 3.0,
                        post: float = 15.0, wait: int = 1) -> PingResult:
    """
    Run a steady ping while `action()` injects a fault mid-stream and estimate
    the data-plane outage as the largest gap between successful replies.

    `action` is invoked once, `pre` seconds into the run, and must NOT restore
    the fault (a fixture is responsible for restoration).
    """
    duration = pre + post
    count = int(duration / interval) + 5
    cmd = [
        "docker", "exec", cname(ep_src.container),
        "ip", "netns", "exec", ep_src.netns,
        "ping", "-D", "-i", str(interval), "-W", str(wait), "-c", str(count), dst_ip,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        time.sleep(pre)
        action()
        out, _ = proc.communicate(timeout=duration + 60)
    finally:
        if proc.poll() is None:
            proc.kill()
            out, _ = proc.communicate()

    res = _parse_stats(out)
    ts = [float(m) for m in
          re.findall(r"^\[(\d+\.\d+)\][^\n]*bytes from", out, re.M)]
    if len(ts) >= 2:
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        res.outage_s = max(0.0, max(gaps) - interval)
    elif res.received == 0:
        res.outage_s = None  # never recovered
    else:
        res.outage_s = 0.0
    return res


# --------------------------------------------------------------------------- #
# iperf3 multi-flow helpers (ECMP / redundancy under real traffic)
# --------------------------------------------------------------------------- #
#
# NOTE on bandwidth: these containers forward in software, so keep the per-flow
# rate modest (a few Mbit/s). Pushing high rates causes host-side drops that are
# unrelated to the fabric and would mask the convergence behaviour we measure.

@dataclass
class FlowResult:
    name: str
    sent: int
    lost: int
    loss_pct: float


def _iperf_udp_cmd(ep_src: Endpoint, dst_ip: str, bitrate: str,
                   duration: float, streams: int, length: int):
    return [
        "docker", "exec", cname(ep_src.container),
        "ip", "netns", "exec", ep_src.netns,
        "iperf3", "-c", dst_ip, "-u", "-b", bitrate, "-P", str(streams),
        "-l", str(length), "-t", str(duration), "--json",
    ]


def _parse_iperf_json(name: str, raw: str) -> FlowResult:
    import json
    try:
        data = json.loads(raw)
        s = data["end"]["sum"]
        return FlowResult(name, int(s.get("packets", 0)),
                          int(s.get("lost_packets", 0)),
                          float(s.get("lost_percent", 100.0)))
    except Exception:
        return FlowResult(name, 0, 0, 100.0)


def run_flows(flows: list[tuple[str, str]], bitrate: str = "3M",
              duration: float = 20.0, streams: int = 4, length: int = 1200,
              action=None, action_delay: float = 6.0) -> list[FlowResult]:
    """
    Launch each (src_endpoint_key, dst_endpoint_key) flow concurrently as a UDP
    iperf3 test (each flow uses `streams` parallel sub-streams for ECMP entropy).
    Optionally invoke `action()` `action_delay` seconds in (e.g. fail a gateway).
    Returns per-flow loss results once all complete.
    """
    procs = []
    for src_key, dst_key in flows:
        src, dst = ENDPOINTS[src_key], ENDPOINTS[dst_key]
        cmd = _iperf_udp_cmd(src, dst.ip, bitrate, duration, streams, length)
        procs.append((f"{src_key}->{dst_key}",
                      subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)))
    if action is not None:
        time.sleep(action_delay)
        action()
    results = []
    for name, p in procs:
        out, _ = p.communicate(timeout=duration + 60)
        results.append(_parse_iperf_json(name, out))
    return results


def dcgw_dci_out_packets(node: str) -> int:
    """Total packets a DCGW has forwarded out of its DCI-facing ports."""
    total = 0
    for port in DCGW_DCI_PORTS:
        out = srl(node, f"info from state interface {port} statistics out-packets")
        m = re.search(r"out-packets\s+(\d+)", out)
        if m:
            total += int(m.group(1))
    return total


def all_dcgws_meshed() -> bool:
    """True when every DCGW holds BOTH an LDP and an SR-ISIS tunnel to each
    remote-DC DCGW loopback.

    A ping canary alone cannot prove the fabric fully recovered: because every
    service fails over to its peer gateway, an isolated DCGW (ports left down) or
    an unrestored transport stays invisible to end-to-end pings. This control-
    plane check catches both, so a botched teardown surfaces instead of silently
    leaving the lab degraded.
    """
    for node in ALL_DCGWS:
        tunnels = dci_tunnels(node)
        for loopback in remote_dcgw_loopbacks(node):
            row = tunnels.get(f"{loopback}/32", {})
            if not all(t in row for t in WAN_TRANSPORTS):
                return False
    return True


def wait_until_healthy(timeout: float = 180.0, settle: int = 2) -> bool:
    """Poll until the canary cross-DC L2 flow is lossless AND all four DCGWs hold
    both WAN transports to every remote DCGW loopback (post-restore sanity)."""
    deadline = time.time() + timeout
    good = 0
    while time.time() < deadline:
        if (ping(CANARY_SRC, CANARY_DST.ip, count=2, interval=0.3).loss_pct == 0.0
                and all_dcgws_meshed()):
            good += 1
            if good >= settle:
                return True
        else:
            good = 0
        time.sleep(2)
    return False


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session", autouse=True)
def lab_ready():
    """Skip the whole suite early if the lab is not deployed."""
    clients = ["mh-dc1", "mh-dc2", "sh-dc1", "sh-dc2", "mh-dc1b", "mh-dc2b"]
    missing = [n for n in ALL_DCGWS + clients
               if cname(n) not in _docker_names()]
    if missing:
        pytest.skip(f"lab not deployed - missing containers: {missing}")
    yield


@pytest.fixture(scope="session")
def bgp_peers_fabric() -> list[dict]:
    """Fabric-wide BGP neighbor state (one fcli query, shared by the suite)."""
    return fcli_json(["bgp-peers"])


@pytest.fixture(scope="session")
def evpn_nexthops_by_node() -> dict[str, dict[int, set[str]]]:
    """Fabric-wide EVPN next-hops grouped by node and route type
    (2=MAC/IP, 3=inclusive-multicast, 5=IP-prefix), collected via fcli."""
    by_node: dict[str, dict[int, set[str]]] = {}
    for rt in (2, 3, 5):
        for row in fcli_json(["bgp-rib", "-r", "evpn", "-t", str(rt)]):
            node, nh = row.get("Node"), row.get("next-hop")
            if node and nh:
                by_node.setdefault(node, {}).setdefault(rt, set()).add(nh)
    return by_node


@pytest.fixture
def restore_direct_mesh():
    """Re-enable the DC1 direct-mesh ports after a failover test."""
    yield
    for node in DC1_DCGWS:
        set_ports(node, DIRECT_MESH_PORTS[node], "enable")
    assert wait_until_healthy(), "fabric did not recover after restoring direct mesh"


@pytest.fixture
def restore_dcgw():
    """Re-enable every port on any DCGW a test isolated."""
    isolated: list[str] = []
    yield isolated
    for node in isolated:
        set_ports(node, DCGW_ALL_PORTS, "enable")
    if isolated:
        assert wait_until_healthy(), "fabric did not recover after restoring DCGW(s)"


@pytest.fixture
def restore_wan_transport():
    """Re-enable any WAN MPLS transport (ldp/sr-isis) a test disabled."""
    disabled: list[str] = []
    yield disabled
    for transport in disabled:
        enable_wan_transport(transport)
    if disabled:
        assert wait_until_healthy(), "fabric did not recover after restoring WAN transport"
