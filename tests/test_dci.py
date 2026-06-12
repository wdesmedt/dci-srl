# © 2025 Nokia
# Licensed under the BSD 3-Clause License
# SPDX-License-Identifier: BSD-3-Clause
"""
Ad-hoc validation suite for the DCI validated design.

Run from this directory (lab must be deployed):

    python3 -m venv .venv && . .venv/bin/activate
    pip install -r requirements.txt
    pytest -v                         # everything
    pytest -v -m connectivity         # just the steady-state checks
    pytest -v -m "not disruptive"     # skip anything that injects a fault
    pytest -v -m convergence -s       # failover/redundancy + outage timing

Disruptive tests always restore the topology in fixture teardown.
"""
from __future__ import annotations

import time

import pytest

from conftest import (
    DC1_DCGWS,
    DC1_LEAVES,
    DC2_LEAVES,
    DC_DCGWS,
    DC_LEAVES,
    ES_LEAVES,
    DC_VTEP_SUBNET,
    LOCAL_DCGW_VTEPS,
    DIRECT_MESH_PORTS,
    DCGW_ALL_PORTS,
    DIAGONAL_DCGW_PAIRS,
    ENDPOINTS,
    ALL_DCGWS,
    WAN_TRANSPORTS,
    dci_tunnels,
    dcgw_dci_out_packets,
    disable_wan_transport,
    l3dci_ipv4_rib,
    measure_convergence,
    ping,
    remote_dcgw_loopbacks,
    run_flows,
    set_ports,
    srl,
    wait_until_healthy,
)

# outage budget (seconds) - generous; the test also prints the measured value
WAN_FAILOVER_BUDGET = 20.0


# --------------------------------------------------------------------------- #
# 0. control-plane precondition
# --------------------------------------------------------------------------- #

@pytest.mark.connectivity
def test_control_plane_established(bgp_peers_fabric):
    """Every DCGW BGP session (iBGP WAN mesh + eBGP fabric) is established.

    Uses a single fabric-wide `fcli bgp-peers` query instead of per-node CLI
    scraping, so one assertion covers all four gateways at once.
    """
    rows = [r for r in bgp_peers_fabric
            if r.get("Node") in ALL_DCGWS
            and r.get("group") in ("wan-ibgp", "dc-fabric")]
    assert rows, "fcli bgp-peers returned no DCGW wan-ibgp/dc-fabric sessions"
    bad = [f"{r.get('Node')} {r.get('1_peer')} [{r.get('group')}]: {r.get('state')}"
           for r in rows if r.get("state") != "established"]
    assert not bad, "non-established DCGW session(s):\n" + "\n".join(bad)


# --------------------------------------------------------------------------- #
# 1. steady-state connectivity (L2 + L3, multi-homed + single-homed)
# --------------------------------------------------------------------------- #

# cross-DC flows: (source endpoint key, destination endpoint key)
L2_FLOWS = [
    ("mh1-l2a", "mh2-l2a"),   # MH -> MH
    ("sh1-l2", "sh2-l2"),     # SH -> SH
    ("mh1-l2b", "sh2-l2"),    # MH -> SH
]
L3_FLOWS = [
    ("mh1-l3a", "mh2-l3a"),   # MH -> MH (routed across DCI)
    ("sh1-l3", "sh2-l3"),     # SH -> SH
    ("mh1-l3b", "sh2-l3"),    # MH -> SH
]


@pytest.mark.connectivity
@pytest.mark.parametrize("src,dst", L2_FLOWS, ids=[f"{s}->{d}" for s, d in L2_FLOWS])
def test_l2_dci_connectivity(src, dst):
    """L2 DCI: stretched bridge-domain reachability across both DCs."""
    res = ping(ENDPOINTS[src], ENDPOINTS[dst].ip, count=5)
    assert res.loss_pct == 0.0, f"{src}->{dst} loss={res.loss_pct}%"


@pytest.mark.connectivity
@pytest.mark.parametrize("src,dst", L3_FLOWS, ids=[f"{s}->{d}" for s, d in L3_FLOWS])
def test_l3_dci_connectivity(src, dst):
    """L3 DCI: routed inter-subnet reachability across both DCs."""
    res = ping(ENDPOINTS[src], ENDPOINTS[dst].ip, count=5)
    assert res.loss_pct == 0.0, f"{src}->{dst} loss={res.loss_pct}%"


# --------------------------------------------------------------------------- #
# 1b. DCI isolation: remote-DC destinations are always re-originated by the
#     LOCAL DCGW pair (no raw VXLAN route leaks end-to-end across the WAN)
# --------------------------------------------------------------------------- #

_LEAF_DC = ([(leaf, 1) for leaf in DC1_LEAVES]
            + [(leaf, 2) for leaf in DC2_LEAVES])


@pytest.mark.connectivity
@pytest.mark.parametrize("leaf,dc", _LEAF_DC, ids=[l for l, _ in _LEAF_DC])
def test_remote_dest_via_local_dcgw(leaf, dc, evpn_nexthops_by_node):
    """
    On every leaf, all EVPN next-hops must live in the LOCAL DC's VTEP subnet:
    intra-DC routes point at local leaves, and remote-DC destinations are
    re-originated by the local DCGW pair (next-hop = a local DCGW VTEP). A
    next-hop in the *remote* DC's subnet would mean a raw VXLAN route leaked
    end-to-end across the DCI instead of being stitched at the gateways.

    Covers BOTH services: type-2 (L2 DCI / MAC) and type-5 (L3 DCI / IP-prefix)
    remote routes must each be present AND arrive via the local DCGW pair.

    Next-hops come from a single fabric-wide `fcli bgp-rib -r evpn` query (which
    includes non-best paths, so a leaked route cannot hide), grouped per node.
    """
    remote_dc = 2 if dc == 1 else 1
    local = LOCAL_DCGW_VTEPS[dc]
    by_type = evpn_nexthops_by_node.get(leaf, {})     # {2: {...}, 3: {...}, 5: {...}}
    nh = set().union(*by_type.values()) if by_type else set()
    assert nh, f"{leaf}: no EVPN routes returned by fcli"

    # 1) no destination may resolve via a REMOTE-DC VTEP (would be an end-to-end leak)
    leaked = sorted(h for h in nh if h.startswith(DC_VTEP_SUBNET[remote_dc]))
    assert not leaked, (
        f"{leaf} (DC{dc}) sees remote-DC VTEP next-hop(s) {leaked} - remote "
        f"destinations must be re-originated by the local DCGWs "
        f"{sorted(local)}, not leaked end-to-end across the DCI"
    )

    # 2) both L2 (type-2) and L3 (type-5) DCI routes are present AND stitched via
    #    a local DCGW (so the test cannot pass on a half-provisioned fabric)
    assert by_type.get(2, set()) & local, (
        f"{leaf}: no L2 DCI (type-2/MAC) route via a local DCGW {sorted(local)}; "
        f"type-2 next-hops seen: {sorted(by_type.get(2, set()))}"
    )
    assert by_type.get(5, set()) & local, (
        f"{leaf}: no L3 DCI (type-5/IP-prefix) route via a local DCGW {sorted(local)}; "
        f"type-5 next-hops seen: {sorted(by_type.get(5, set()))}"
    )


# --------------------------------------------------------------------------- #
# 1c. L3 host-route: anti-trombone inside the DC, blocked across the DCI
# --------------------------------------------------------------------------- #
#
# `arp evpn advertise dynamic interface-less-routing` on the leaf IRBs advertises
# each learned host as an EVPN MAC/IP route carrying the IP-VRF interface-less
# label+RT, so EVERY leaf/DCGW in the host's DC installs a `bgp-evpn-ifl-host` /32
# pointing directly at the owning leaf's VTEP (no trombone through the anycast
# /24). On a multi-homed ES, `advertise-ifl-host-ad-routes` makes BOTH ES leaves
# originate the host route, so the /32 aliases across both their VTEPs (inter-
# subnet load-balancing). For scaling, those host-routes must NOT cross the DCI:
# they carry the VXLAN tunnel-encap community (dropped on dcgw-wan-export-dcX
# statement 10) plus a defense-in-depth /32 l3vpn reject (statement 8 +
# host-routes-l3dci set), so only the covering /24 crosses - the remote DC never
# sees a host /32.

# (host endpoint key, cross-DC source used to warm its ARP/host-route, multi-homed?)
L3_HOST_CHECKS = [
    ("mh2-l3a", "mh1-l3a", True),    # DC2 multi-homed host (mh-dc2 ES on leaf5/6)
    ("mh1-l3a", "mh2-l3a", True),    # DC1 multi-homed host (mh-dc1 ES on leaf1/2)
    ("sh2-l3", "sh1-l3", False),     # DC2 single-homed host
    ("sh1-l3", "sh2-l3", False),     # DC1 single-homed host
]


@pytest.mark.connectivity
@pytest.mark.parametrize("host,pinger,multihomed", L3_HOST_CHECKS,
                         ids=[h for h, _, _ in L3_HOST_CHECKS])
def test_l3_host_route_scoped_to_local_dc(host, pinger, multihomed):
    """A host /32 is installed on every node of its OWN DC (anti-trombone) - for a
    multi-homed host via BOTH its ES leaf VTEPs (IP-aliasing / load-balancing) -
    but is blocked across the DCI: the remote DC holds only the covering /24."""
    ep = ENDPOINTS[host]
    home = ep.dc
    remote = 2 if home == 1 else 1
    host32 = f"{ep.ip}/32"
    subnet24 = ".".join(ep.ip.split(".")[:3]) + ".0/24"
    home_nodes = set(DC_LEAVES[home]) | set(DC_DCGWS[home])
    remote_nodes = set(DC_LEAVES[remote]) | set(DC_DCGWS[remote])
    # the leaves the host is directly attached to reach it via the connected
    # anycast /24 + local ARP (no /32); every OTHER home-DC node must install the
    # bgp-evpn-ifl host /32 - those are the nodes where a trombone could occur.
    owners = set(ES_LEAVES[ep.container])
    expect_nodes = home_nodes - owners

    # warm the host's ARP entry (=> /32 host-route) with a cross-DC ping
    assert ping(ENDPOINTS[pinger], ep.ip, count=3).ok, \
        f"warm-up ping {pinger}->{host} failed"

    def active(rib, node):
        return {r["Prefix"] for r in rib
                if r.get("Node") == node and r.get("Act") == "yes"}

    def present(rib, node):
        return {r["Prefix"] for r in rib if r.get("Node") == node}

    def nexthops(rib, node):
        return {
            nh for r in rib
            if r.get("Node") == node and r.get("Prefix") == host32
            and r.get("Act") == "yes"
            for nh in (r.get("next-hop") or [])
        }

    # poll the fabric IPv4 RIB until the /32 has propagated to every node that
    # should hold it (multi-homed => via both ES VTEPs)
    def ready(rib):
        return all(host32 in active(rib, n) for n in expect_nodes) and (
            not multihomed
            or all(len(nexthops(rib, n)) >= len(owners) for n in expect_nodes))

    rib = l3dci_ipv4_rib()
    for _ in range(6):
        if ready(rib):
            break
        time.sleep(2)
        rib = l3dci_ipv4_rib()

    # 1) anti-trombone: every non-attached node in the host's DC installs the /32
    #    directly (ingress there goes straight to the owning leaf, no hairpin)
    missing = sorted(n for n in expect_nodes if host32 not in active(rib, n))
    assert not missing, (
        f"{host32}: host-route absent on home-DC node(s) {missing} - ingress "
        f"there would trombone via the anycast {subnet24}"
    )

    # 2) that /32 is resolved INSIDE the DC (VXLAN to a local leaf), never pulled
    #    back over the WAN (ldp/sr-isis == an MPLS DCI tunnel)
    via_wan = sorted({
        r["Node"] for r in rib
        if r.get("Prefix") == host32 and r.get("Node") in expect_nodes
        and any(str(i).startswith(("ldp:", "sr-isis:")) for i in (r.get("itf") or []))
    })
    assert not via_wan, f"{host32} resolved over the WAN on home-DC node(s) {via_wan}"

    # 3) scaling: the /32 must NOT cross the DCI (absent on every remote-DC node,
    #    even as a non-best path), while the covering /24 IS present there
    leaked = sorted(n for n in remote_nodes if host32 in present(rib, n))
    assert not leaked, (
        f"{host32} leaked across the DCI to remote-DC node(s) {leaked}; only "
        f"{subnet24} may cross the WAN"
    )
    no_subnet = sorted(n for n in remote_nodes if subnet24 not in active(rib, n))
    assert not no_subnet, \
        f"remote-DC node(s) missing the covering {subnet24}: {no_subnet}"

    # 4) multi-homing load-balancing: a multi-homed host /32 must alias across ALL
    #    its ES leaf VTEPs (advertise-ifl-host-ad-routes), so every other in-DC
    #    node resolves it via >=2 next-hops - otherwise inter-subnet traffic to the
    #    host is pinned to a single ES leaf.
    if multihomed:
        not_aliased = sorted(n for n in expect_nodes
                             if len(nexthops(rib, n)) < len(owners))
        assert not not_aliased, (
            f"{host32} not load-balanced across the ES ({len(owners)} VTEPs) on "
            f"home-DC node(s) {not_aliased}: "
            + "; ".join(f"{n}={sorted(nexthops(rib, n))}" for n in expect_nodes)
        )


# --------------------------------------------------------------------------- #
# 2. transport path: DIRECT mesh preferred in steady state
# --------------------------------------------------------------------------- #

@pytest.mark.connectivity
def test_direct_path_preferred():
    """Both transports' tunnels to the remote-DC DCGW loopbacks ride the direct mesh."""
    # dcgw1 -> dcgw3 (192.0.3.153): each of the ldp + sr-isis tunnels (metric 10)
    # must egress a direct-mesh port (ethernet-1/3..5), not a P-facing WAN uplink.
    tunnels = dci_tunnels("dcgw1")
    row = tunnels.get("192.0.3.153/32", {})
    assert row, f"no tunnels to dcgw3 loopback:\n{srl('dcgw1', 'show network-instance default tunnel-table all')}"
    mesh = [f"{p}." for p in DIRECT_MESH_PORTS["dcgw1"]]  # e.g. 'ethernet-1/3.'
    for transport in WAN_TRANSPORTS:
        port = row.get(transport)
        assert port, f"no {transport} tunnel to dcgw3 loopback (saw {row})"
        assert any(port.startswith(p) for p in mesh), (
            f"{transport} tunnel to dcgw3 not on a direct-mesh port "
            f"(expected one of {mesh}): {port}"
        )


@pytest.mark.connectivity
@pytest.mark.parametrize("node", ALL_DCGWS)
def test_wan_transports_both_present(node):
    """Each DCGW must hold BOTH an LDP and an SR-ISIS tunnel to every remote-DC
    DCGW loopback, i.e. the two MPLS transports run in parallel and either can
    carry DCI services (validated under fault by test_wan_transport_failover)."""
    tunnels = dci_tunnels(node)
    for loopback in remote_dcgw_loopbacks(node):
        row = tunnels.get(f"{loopback}/32", {})
        missing = [t for t in WAN_TRANSPORTS if t not in row]
        assert not missing, (
            f"{node}: missing {missing} tunnel(s) to remote DCGW {loopback} "
            f"(present: {sorted(row)}); both LDP and SR-ISIS must be programmed"
        )


# --------------------------------------------------------------------------- #
# 3. DIRECT-connect failure -> traffic survives over the WAN (P/PE) core
# --------------------------------------------------------------------------- #

def _disable_dc1_direct_mesh():
    for node in DC1_DCGWS:
        set_ports(node, DIRECT_MESH_PORTS[node], "disable")


@pytest.mark.disruptive
@pytest.mark.convergence
@pytest.mark.parametrize("src,dst", [("mh1-l2a", "mh2-l2a")], ids=["l2"])
def test_wan_failover_l2(src, dst, restore_direct_mesh, record_property):
    """Fail the DC1 direct mesh; L2 DCI must reconverge over the WAN core."""
    res = measure_convergence(ENDPOINTS[src], ENDPOINTS[dst].ip,
                              action=_disable_dc1_direct_mesh, post=15.0)
    record_property("outage_seconds", res.outage_s)
    print(f"\n[L2 WAN failover] loss={res.loss_pct}% outage~={res.outage_s}s")
    assert res.received > 0 and res.outage_s is not None, "no recovery over WAN path"
    assert res.outage_s < WAN_FAILOVER_BUDGET, f"outage {res.outage_s}s exceeds budget"
    # confirm steady connectivity while still on the WAN path
    assert ping(ENDPOINTS[src], ENDPOINTS[dst].ip, count=5).loss_pct == 0.0


@pytest.mark.disruptive
@pytest.mark.convergence
@pytest.mark.parametrize("src,dst", [("mh1-l3a", "mh2-l3a")], ids=["l3"])
def test_wan_failover_l3(src, dst, restore_direct_mesh, record_property):
    """Fail the DC1 direct mesh; L3 DCI must reconverge over the WAN core."""
    res = measure_convergence(ENDPOINTS[src], ENDPOINTS[dst].ip,
                              action=_disable_dc1_direct_mesh, post=15.0)
    record_property("outage_seconds", res.outage_s)
    print(f"\n[L3 WAN failover] loss={res.loss_pct}% outage~={res.outage_s}s")
    assert res.received > 0 and res.outage_s is not None, "no recovery over WAN path"
    assert res.outage_s < WAN_FAILOVER_BUDGET, f"outage {res.outage_s}s exceeds budget"
    assert ping(ENDPOINTS[src], ENDPOINTS[dst].ip, count=5).loss_pct == 0.0


# --------------------------------------------------------------------------- #
# 4. ECMP: L3 DCI flows must hash across BOTH local DCGWs
# --------------------------------------------------------------------------- #
#
# Each flow uses several parallel UDP sub-streams (distinct L4 ports => distinct
# VXLAN entropy), and the flows originate from both DC1 Ethernet-Segments
# (mh-dc1 on leaf1/2 + mh-dc1b on leaf3/4) plus the single-homed client.
#
# NOTE on L2 vs L3 gateway behaviour:
#   * L3 DCI uses EVPN-IFL/IP-VPN: the remote prefix is reachable via both DCGWs
#     as independent ECMP next-hops, so flows load-balance across both gateways
#     (active/active) -> asserted below.
#   * L2 DCI uses the documented anycast model (shared RD + inclusive-mcast
#     originating-ip): leaves receive equivalent MAC routes from both gateways
#     and BGP selects ONE (active/standby for unicast, active/active for BUM).
#     Redundancy is therefore validated by failover (section 5), not by hashing.
#     Per-DCGW *load-sharing* for L2 is instead achieved at service granularity:
#     a 2nd stretched BD (BD-B, vlan 110) is pinned to dcgw2/dcgw4 by de-preferring
#     it on the other gateway, applied consistently on BOTH DCI planes -- AS-path
#     prepend on the eBGP fabric and lower local-preference on the iBGP WAN core --
#     so BD-A rides dcgw1/3 and BD-B rides dcgw2/4 for both the local- and remote-DC
#     egress (asserted by test_l2_per_service_gateway_pinning).

# (src_endpoint, dst_endpoint) - each dst is a distinct iperf3 server (no contention)
L2_MULTIFLOW = [
    ("mh1-l2a", "mh2-l2a"), ("mh1-l2b", "mh2-l2b"),
    ("mh1b-l2a", "mh2b-l2a"), ("mh1b-l2b", "mh2b-l2b"),
    ("sh1-l2", "sh2-l2"),
]
L3_MULTIFLOW = [
    ("mh1-l3a", "mh2-l3a"), ("mh1-l3b", "mh2-l3b"),
    ("mh1b-l3a", "mh2b-l3a"), ("mh1b-l3b", "mh2b-l3b"),
    ("sh1-l3", "sh2-l3"),
]
# BD-B (vlan 110) cross-DC flows - the 2nd stretched bridge-domain (pinned dcgw2/4)
L2_BDB_MULTIFLOW = [
    ("mh1-l2c", "mh2-l2c"), ("mh1-l2d", "mh2-l2d"),
    ("sh1-l2b", "sh2-l2b"),
]

# keep per-flow bandwidth modest: these are software-forwarding containers, so a
# high rate would cause host-side drops unrelated to the fabric (see conftest note)
FLOW_BITRATE = "1M"
FLOW_STREAMS = 4


@pytest.mark.ecmp
def test_l3_ecmp_spreads_across_both_dcgws(record_property):
    """Multiple L3 DCI flows (DC1->DC2) must be forwarded by BOTH DC1 gateways."""
    b1, b2 = dcgw_dci_out_packets("dcgw1"), dcgw_dci_out_packets("dcgw2")
    run_flows(L3_MULTIFLOW, bitrate=FLOW_BITRATE, duration=8.0, streams=FLOW_STREAMS)
    d1 = dcgw_dci_out_packets("dcgw1") - b1
    d2 = dcgw_dci_out_packets("dcgw2") - b2
    total = d1 + d2
    record_property("dcgw1_fwd_pkts", d1)
    record_property("dcgw2_fwd_pkts", d2)
    print(f"\n[ECMP] dcgw1 forwarded={d1} dcgw2 forwarded={d2} "
          f"(share {d1/total:.0%}/{d2/total:.0%})" if total else "\n[ECMP] no traffic")
    assert total > 2000, f"little/no DCI traffic observed (total={total})"
    # both gateways must carry a meaningful share => flows are hashed across them
    assert min(d1, d2) > 0.15 * total, (
        f"traffic not spread across both DCGWs: dcgw1={d1} dcgw2={d2}"
    )


# per-service L2 load-sharing: BD-A (vlan 100) pins to dcgw1, BD-B (vlan 110) to dcgw2
L2_BDA_FLOWS = [("mh1-l2a", "mh2-l2a"), ("sh1-l2", "sh2-l2")]
L2_BDB_FLOWS = [("mh1-l2c", "mh2-l2c"), ("sh1-l2b", "sh2-l2b")]


@pytest.mark.ecmp
def test_l2_per_service_gateway_pinning(record_property):
    """
    The two stretched L2 bridge-domains must use DIFFERENT gateways of the DC1
    pair: BD-A (vlan 100) -> dcgw1, BD-B (vlan 110) -> dcgw2. Steered by de-preferring
    each BD on the non-preferred gateway on both planes (AS-path prepend on the eBGP
    fabric, lower local-preference on the iBGP WAN). Verified from the DCI egress
    counters; the same pinning makes BD-B egress the remote DC via dcgw4.
    """
    def dominant(flows):
        b1, b2 = dcgw_dci_out_packets("dcgw1"), dcgw_dci_out_packets("dcgw2")
        run_flows(flows, bitrate=FLOW_BITRATE, duration=8.0, streams=FLOW_STREAMS)
        d1 = dcgw_dci_out_packets("dcgw1") - b1
        d2 = dcgw_dci_out_packets("dcgw2") - b2
        return ("dcgw1" if d1 >= d2 else "dcgw2"), d1, d2

    gw_a, a1, a2 = dominant(L2_BDA_FLOWS)
    gw_b, b1, b2 = dominant(L2_BDB_FLOWS)
    record_property("bd_a_gw", gw_a)
    record_property("bd_b_gw", gw_b)
    print(f"\n[L2 per-service] BD-A(vlan100)->{gw_a} (dcgw1={a1} dcgw2={a2}) ; "
          f"BD-B(vlan110)->{gw_b} (dcgw1={b1} dcgw2={b2})")
    assert a1 + a2 > 2000 and b1 + b2 > 2000, "little/no L2 DCI traffic observed"
    # each service is steered onto its own gateway, and they differ
    assert gw_a == "dcgw1", f"BD-A expected on dcgw1, dominant was {gw_a}"
    assert gw_b == "dcgw2", f"BD-B expected on dcgw2, dominant was {gw_b}"
    assert a1 > 3 * a2, f"BD-A not pinned to dcgw1: dcgw1={a1} dcgw2={a2}"
    assert b2 > 3 * b1, f"BD-B not pinned to dcgw2: dcgw1={b1} dcgw2={b2}"


# --------------------------------------------------------------------------- #
# 5. DCGW redundancy under multi-flow load: isolate a GW, flows survive via peer
# --------------------------------------------------------------------------- #

@pytest.mark.disruptive
@pytest.mark.convergence
@pytest.mark.parametrize("flows", [L2_MULTIFLOW, L3_MULTIFLOW], ids=["l2", "l3"])
def test_gw_redundancy_multiflow(flows, restore_dcgw, record_property):
    """
    Run many hashed flows, isolate dcgw1 mid-stream, and confirm every flow
    survives via dcgw2 (anycast for L2 / IP-VPN ECMP for L3) with low loss.
    """
    victim = "dcgw1"
    duration = 25.0

    def _isolate():
        restore_dcgw.append(victim)
        set_ports(victim, DCGW_ALL_PORTS, "disable")

    results = run_flows(flows, bitrate=FLOW_BITRATE, duration=duration,
                        streams=FLOW_STREAMS, action=_isolate, action_delay=6.0)
    losses = [r.loss_pct for r in results]
    worst, avg = max(losses), sum(losses) / len(losses)
    for r in results:
        print(f"\n  {r.name}: loss={r.loss_pct:.2f}% ({r.lost}/{r.sent})")
    # rough convergence proxy from the worst flow (UDP datagram-rate based)
    record_property("worst_loss_pct", worst)
    record_property("avg_loss_pct", avg)
    print(f"\n[GW redundancy, isolate {victim}] worst-flow loss={worst:.2f}% "
          f"avg={avg:.2f}% over {len(results)} hashed flows")
    assert all(r.sent > 0 for r in results), "a flow sent no traffic"
    assert worst < 25.0, f"a flow lost too much during GW failure: {worst:.2f}%"
    assert avg < 10.0, f"average loss across flows too high: {avg:.2f}%"


# --------------------------------------------------------------------------- #
# 6. WAN transport failover: disable one MPLS transport, services ride the other
# --------------------------------------------------------------------------- #
#
# Both LDP and SR-ISIS tunnels exist to every remote loopback (LDP preferred).
# Disabling one transport on every WAN node must transparently move all DCI
# services onto the survivor; the transport is restored (and full connectivity
# reverified) in fixture teardown.

@pytest.mark.disruptive
@pytest.mark.convergence
@pytest.mark.parametrize("transport", list(WAN_TRANSPORTS),
                         ids=[f"disable-{t}" for t in WAN_TRANSPORTS])
def test_wan_transport_failover(transport, restore_wan_transport, record_property):
    """Disable `transport` (ldp|sr-isis) on every WAN node mid-stream; all L2/L3
    DCI flows must reconverge onto the surviving transport, and the tunnel-table
    must then hold ONLY the survivor."""
    survivor = next(t for t in WAN_TRANSPORTS if t != transport)
    flows = L2_MULTIFLOW + L3_MULTIFLOW

    def _disable():
        restore_wan_transport.append(transport)   # ensure restoration even on failure
        disable_wan_transport(transport)

    results = run_flows(flows, bitrate=FLOW_BITRATE, duration=40.0,
                        streams=FLOW_STREAMS, action=_disable, action_delay=8.0)
    losses = [r.loss_pct for r in results]
    worst, avg = max(losses), sum(losses) / len(losses)
    for r in results:
        print(f"\n  {r.name}: loss={r.loss_pct:.2f}% ({r.lost}/{r.sent})")
    record_property("worst_loss_pct", worst)
    record_property("avg_loss_pct", avg)
    print(f"\n[WAN transport failover, disable {transport} -> {survivor}] "
          f"worst-flow loss={worst:.2f}% avg={avg:.2f}% over {len(results)} flows")
    assert all(r.sent > 0 for r in results), "a flow sent no traffic"
    assert worst < 25.0, f"a flow lost too much moving to {survivor}: {worst:.2f}%"
    assert avg < 10.0, f"average loss too high moving to {survivor}: {avg:.2f}%"

    # the disabled transport must be gone and only the survivor present on a sample DCGW
    tunnels = dci_tunnels("dcgw1")
    assert tunnels, "no DCI tunnels in the table after transport disable"
    for loopback in remote_dcgw_loopbacks("dcgw1"):
        row = tunnels.get(f"{loopback}/32", {})
        assert survivor in row, f"survivor {survivor} tunnel to {loopback} missing: {row}"
        assert transport not in row, (
            f"disabled transport {transport} still present to {loopback}: {row}"
        )


# --------------------------------------------------------------------------- #
# 7. Diagonal double DCGW failure: one gateway per DC fails simultaneously
# --------------------------------------------------------------------------- #
#
# The two DCGW pairs form a 2x2 grid (DC1={dcgw1,dcgw2}, DC2={dcgw3,dcgw4}).
# A "diagonal" failure isolates one gateway in EACH DC at the same time
# ({dcgw1,dcgw4} or {dcgw2,dcgw3}). Because BD-A is pinned to dcgw1/dcgw3 and
# BD-B to dcgw2/dcgw4, each diagonal kills the *primary* gateway of one BD in DC1
# and of the other BD in DC2 - so both DCs must fail those services over to their
# surviving gateway at once. Every cross-DC flow must survive, and once the
# gateways are restored full connectivity must return.

@pytest.mark.disruptive
@pytest.mark.convergence
@pytest.mark.parametrize("victims", DIAGONAL_DCGW_PAIRS,
                         ids=["+".join(p) for p in DIAGONAL_DCGW_PAIRS])
def test_diagonal_dcgw_double_failure(victims, restore_dcgw, record_property):
    """Isolate one DCGW per DC (a diagonal of the gateway grid); BD-A, BD-B and L3
    flows must all survive via the surviving gateway in each DC, and connectivity
    must be fully restored once the gateways come back."""
    # cover both bridge-domains (so a pinned service is forced to fail over) + L3,
    # across both DC1 Ethernet-Segments and the single-homed clients
    flows = [
        ("mh1-l2a", "mh2-l2a"), ("mh1b-l2a", "mh2b-l2a"), ("sh1-l2", "sh2-l2"),
        ("mh1-l2c", "mh2-l2c"), ("sh1-l2b", "sh2-l2b"),
        ("mh1-l3a", "mh2-l3a"), ("mh1b-l3a", "mh2b-l3a"), ("sh1-l3", "sh2-l3"),
    ]
    duration = 35.0

    def _isolate():
        for v in victims:
            restore_dcgw.append(v)          # safety-net restoration in fixture
        for v in victims:
            set_ports(v, DCGW_ALL_PORTS, "disable")

    results = run_flows(flows, bitrate=FLOW_BITRATE, duration=duration,
                        streams=FLOW_STREAMS, action=_isolate, action_delay=8.0)
    losses = [r.loss_pct for r in results]
    worst, avg = max(losses), sum(losses) / len(losses)
    for r in results:
        print(f"\n  {r.name}: loss={r.loss_pct:.2f}% ({r.lost}/{r.sent})")
    record_property("worst_loss_pct", worst)
    record_property("avg_loss_pct", avg)
    print(f"\n[diagonal failure {'+'.join(victims)}] worst-flow loss={worst:.2f}% "
          f"avg={avg:.2f}% over {len(results)} flows")
    assert all(r.sent > 0 for r in results), "a flow sent no traffic"
    # both DCs lose a gateway at once, so allow a slightly larger budget than a
    # single-GW failure, but every flow must still survive via its DC's peer GW
    assert worst < 30.0, f"a flow lost too much during diagonal failure: {worst:.2f}%"
    assert avg < 12.0, f"average loss across flows too high: {avg:.2f}%"

    # explicitly verify connectivity is properly RESTORED once the GWs come back
    for v in victims:
        set_ports(v, DCGW_ALL_PORTS, "enable")
    assert wait_until_healthy(), "fabric did not recover after restoring the diagonal GWs"
    post = [(s, d) for s, d in flows]
    bad = [f"{s}->{d}={r}%" for (s, d) in post
           for r in [ping(ENDPOINTS[s], ENDPOINTS[d].ip, count=4).loss_pct] if r != 0.0]
    assert not bad, "flows not fully restored after GW recovery: " + ", ".join(bad)
