"""Adjudication flow tests: routing, cycles, validation, read-only storage.

Run with: python tests/test_adjudication_flows.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.store import store  # noqa: E402

client = TestClient(app)


def setup_function():
    store.clear()


ELIGIBILITY = {
    "rules": [
        {
            "id": "adult-allow",
            "priority": 100,
            "conclusion": "allow",
            "expr": {
                "all": [
                    {"field": "age", "op": "ge", "value": 18},
                    {"field": "country", "op": "eq", "value": "NL"},
                ]
            },
        },
        {
            "id": "minor-deny",
            "priority": 100,
            "conclusion": "deny",
            "expr": {"field": "age", "op": "lt", "value": 18},
        },
    ]
}

RISK = {
    "rules": [
        {
            "id": "vip-approve",
            "priority": 10,
            "conclusion": "approve",
            "expr": {"field": "vip", "op": "eq", "value": True},
        },
        {
            "id": "debt-reject",
            "priority": 10,
            "conclusion": "reject",
            "expr": {"field": "debt_ratio", "op": "ge", "value": 0.6},
        },
    ]
}

CONFLICT = {
    "rules": [
        {"id": "c-yes", "priority": 5, "conclusion": "yes",
         "expr": {"field": "x", "op": "ge", "value": 10}},
        {"id": "c-no", "priority": 5, "conclusion": "no",
         "expr": {"field": "x", "op": "ne", "value": 5}},
    ]
}

SCREEN = {
    "rules": [
        {"id": "hit", "priority": 1, "conclusion": "hit",
         "expr": {"field": "a", "op": "eq", "value": 1}},
    ]
}

FULL_FACTS = {"age": 25, "country": "NL", "vip": True, "debt_ratio": 0.1}


def create_policies():
    ids = {}
    for name, policy in [("eligibility", ELIGIBILITY), ("risk", RISK),
                         ("conflict", CONFLICT), ("screen", SCREEN)]:
        r = client.post("/api/v1/policies", json=policy)
        assert r.status_code == 201, r.text
        ids[name] = r.json()["id"]
    return ids


def post_flow(body):
    return client.post("/api/v1/adjudication-flows/evaluate", json=body)


def loan_flow(ids, routes, facts, extra_nodes=None):
    """Two-node flow: eligibility (with ``routes``) -> risk (terminal)."""
    nodes = [
        {"id": "eligibility", "policy_id": ids["eligibility"],
         "routes": routes},
        {"id": "risk", "policy_id": ids["risk"]},
    ]
    nodes.extend(extra_nodes or [])
    return {"start": "eligibility", "nodes": nodes, "facts": facts}


# --------------------------------------------------------------------------- #
# Routing semantics
# --------------------------------------------------------------------------- #

def test_decided_conclusion_route_chains_to_next_node():
    ids = create_policies()
    body = loan_flow(
        ids, [{"on": "decided", "conclusion": "allow", "next": "risk"}],
        FULL_FACTS,
    )
    r = post_flow(body)
    assert r.status_code == 200, r.text
    out = r.json()

    assert out["termination_reason"] == "no_matching_route"
    assert out["final_signature"] == {
        "status": "decided", "conclusion": "approve", "decided_priority": 10}

    trace = out["trace"]
    assert [t["node_id"] for t in trace] == ["eligibility", "risk"]
    assert trace[0]["policy_id"] == ids["eligibility"]
    assert trace[0]["status"] == "decided"
    assert trace[0]["conclusion"] == "allow"
    assert trace[0]["matched_route"] == {
        "on": "decided", "conclusion": "allow", "next": "risk"}
    assert trace[1]["policy_id"] == ids["risk"]
    assert trace[1]["matched_route"] is None
    assert trace[1]["signature"] == out["final_signature"]


def test_unmatched_verdict_terminates_at_first_node():
    ids = create_policies()
    body = loan_flow(
        ids, [{"on": "decided", "conclusion": "allow", "next": "risk"}],
        {"age": 10, "country": "US"},
    )
    out = post_flow(body).json()
    assert out["termination_reason"] == "no_matching_route"
    assert len(out["trace"]) == 1
    assert out["trace"][0]["matched_route"] is None
    assert out["final_signature"] == {
        "status": "decided", "conclusion": "deny", "decided_priority": 100}


def test_unknown_status_routing():
    ids = create_policies()
    body = loan_flow(ids, [{"on": "unknown", "next": "risk"}], {})
    out = post_flow(body).json()
    assert [t["status"] for t in out["trace"]] == ["unknown", "unknown"]
    signature = out["final_signature"]
    assert signature["status"] == "unknown"
    assert signature["decided_priority"] == 10
    assert signature["minimal_missing_field_sets"] == [["debt_ratio", "vip"]]
    assert signature["pending_conclusions"] == ["approve", "reject"]


def test_conflict_status_routing():
    ids = create_policies()
    body = {
        "start": "check",
        "nodes": [
            {"id": "check", "policy_id": ids["conflict"],
             "routes": [{"on": "conflict", "next": "risk"}]},
            {"id": "risk", "policy_id": ids["risk"]},
        ],
        "facts": {"x": 10, "vip": True, "debt_ratio": 0.1},
    }
    out = post_flow(body).json()
    assert [t["node_id"] for t in out["trace"]] == ["check", "risk"]
    assert out["trace"][0]["status"] == "conflict"
    assert out["trace"][0]["signature"]["conflicting_conclusions"] == [
        "no", "yes"]
    assert out["final_signature"]["conclusion"] == "approve"


def test_no_match_status_routing():
    ids = create_policies()
    body = {
        "start": "screen",
        "nodes": [
            {"id": "screen", "policy_id": ids["screen"],
             "routes": [{"on": "no_match", "next": "risk"}]},
            {"id": "risk", "policy_id": ids["risk"]},
        ],
        "facts": {"a": 2, "vip": True, "debt_ratio": 0.1},
    }
    out = post_flow(body).json()
    assert [t["status"] for t in out["trace"]] == ["no_match", "decided"]
    assert out["final_signature"]["conclusion"] == "approve"


def test_decided_route_without_conclusion_matches_any_decision():
    ids = create_policies()
    body = loan_flow(
        ids, [{"on": "decided", "next": "risk"}],
        {"age": 10, "country": "US", "vip": True, "debt_ratio": 0.1},
    )
    out = post_flow(body).json()
    assert [t["node_id"] for t in out["trace"]] == ["eligibility", "risk"]
    assert out["trace"][0]["conclusion"] == "deny"


def test_first_matching_route_wins():
    ids = create_policies()
    extra = [{"id": "fallback", "policy_id": ids["screen"]}]

    specific_first = loan_flow(ids, [
        {"on": "decided", "conclusion": "allow", "next": "risk"},
        {"on": "decided", "next": "fallback"},
    ], FULL_FACTS, extra_nodes=extra)
    out = post_flow(specific_first).json()
    assert [t["node_id"] for t in out["trace"]] == ["eligibility", "risk"]

    catch_all_first = loan_flow(ids, [
        {"on": "decided", "next": "fallback"},
        {"on": "decided", "conclusion": "allow", "next": "risk"},
    ], FULL_FACTS, extra_nodes=extra)
    out = post_flow(catch_all_first).json()
    assert [t["node_id"] for t in out["trace"]] == ["eligibility", "fallback"]


def test_decided_conclusion_mismatch_falls_through_to_next_route():
    ids = create_policies()
    body = loan_flow(ids, [
        {"on": "decided", "conclusion": "allow", "next": "fallback"},
        {"on": "decided", "conclusion": "deny", "next": "risk"},
    ], {"age": 10, "country": "US", "vip": True, "debt_ratio": 0.1},
        extra_nodes=[{"id": "fallback", "policy_id": ids["screen"]}])
    out = post_flow(body).json()
    assert [t["node_id"] for t in out["trace"]] == ["eligibility", "risk"]


def test_three_node_chain_preserves_execution_order():
    ids = create_policies()
    body = {
        "start": "screen",
        "nodes": [
            {"id": "screen", "policy_id": ids["screen"],
             "routes": [{"on": "no_match", "next": "eligibility"}]},
            {"id": "eligibility", "policy_id": ids["eligibility"],
             "routes": [{"on": "decided", "conclusion": "allow",
                         "next": "risk"}]},
            {"id": "risk", "policy_id": ids["risk"]},
        ],
        "facts": {"a": 2, **FULL_FACTS},
    }
    out = post_flow(body).json()
    assert [t["node_id"] for t in out["trace"]] == [
        "screen", "eligibility", "risk"]
    assert out["final_signature"]["conclusion"] == "approve"


def test_same_policy_can_be_reused_by_multiple_nodes():
    ids = create_policies()
    body = {
        "start": "first",
        "nodes": [
            {"id": "first", "policy_id": ids["eligibility"],
             "routes": [{"on": "decided", "conclusion": "allow",
                         "next": "second"}]},
            {"id": "second", "policy_id": ids["eligibility"]},
        ],
        "facts": {"age": 25, "country": "NL"},
    }
    r = post_flow(body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert [t["node_id"] for t in out["trace"]] == ["first", "second"]
    assert all(t["policy_id"] == ids["eligibility"] for t in out["trace"])


def test_facts_default_to_empty_object():
    ids = create_policies()
    body = {"start": "eligibility",
            "nodes": [{"id": "eligibility", "policy_id": ids["eligibility"]}]}
    r = post_flow(body)
    assert r.status_code == 200, r.text
    assert r.json()["trace"][0]["status"] == "unknown"


def test_repeated_requests_are_byte_identical():
    ids = create_policies()
    body = loan_flow(
        ids, [{"on": "decided", "conclusion": "allow", "next": "risk"}],
        FULL_FACTS,
    )
    first = post_flow(body)
    second = post_flow(body)
    assert first.status_code == 200 and second.status_code == 200
    assert first.content == second.content


# --------------------------------------------------------------------------- #
# Cycle detection (reachable subgraph only)
# --------------------------------------------------------------------------- #

def test_cycle_reachable_from_start_returns_422():
    ids = create_policies()
    body = {
        "start": "a",
        "nodes": [
            {"id": "a", "policy_id": ids["screen"],
             "routes": [{"on": "no_match", "next": "b"}]},
            {"id": "b", "policy_id": ids["screen"],
             "routes": [{"on": "no_match", "next": "a"}]},
        ],
        "facts": {},
    }
    r = post_flow(body)
    assert r.status_code == 422
    out = r.json()
    assert out["detail"] == "Flow validation failed"
    assert any("cycle" in e for e in out["errors"])


def test_self_loop_returns_422():
    ids = create_policies()
    body = {
        "start": "a",
        "nodes": [{"id": "a", "policy_id": ids["screen"],
                   "routes": [{"on": "no_match", "next": "a"}]}],
    }
    r = post_flow(body)
    assert r.status_code == 422
    assert any("cycle" in e for e in r.json()["errors"])


def test_cycle_outside_reachable_subgraph_is_allowed():
    ids = create_policies()
    body = {
        "start": "main",
        "nodes": [
            {"id": "main", "policy_id": ids["screen"]},
            {"id": "loop-1", "policy_id": ids["screen"],
             "routes": [{"on": "no_match", "next": "loop-2"}]},
            {"id": "loop-2", "policy_id": ids["screen"],
             "routes": [{"on": "no_match", "next": "loop-1"}]},
        ],
        "facts": {"a": 1},
    }
    r = post_flow(body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert [t["node_id"] for t in out["trace"]] == ["main"]
    assert out["final_signature"] == {
        "status": "decided", "conclusion": "hit", "decided_priority": 1}


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def test_invalid_flow_requests_return_422():
    ids = create_policies()
    node = {"id": "a", "policy_id": ids["screen"]}
    bad_bodies = [
        "nope",
        {},
        {"nodes": [node]},                                           # no start
        {"start": 1, "nodes": [node]},                               # bad start
        {"start": "ghost", "nodes": [node]},                         # unknown start
        {"start": "a"},                                              # no nodes
        {"start": "a", "nodes": []},                                 # empty nodes
        {"start": "a", "nodes": "nope"},                             # nodes not array
        {"start": "a", "nodes": [1]},                                # node not object
        {"start": "a", "nodes": [{"policy_id": ids["screen"]}]},     # no id
        {"start": "a", "nodes": [node, node]},                       # duplicate id
        {"start": "a", "nodes": [{"id": "a"}]},                      # no policy_id
        {"start": "a", "nodes": [{"id": "a", "policy_id": "nope"}]},  # no policy
        {"start": "a", "nodes": [dict(node, routes="nope")]},        # bad routes
        {"start": "a", "nodes": [dict(node, routes=[1])]},           # bad route
        {"start": "a", "nodes": [dict(node, routes=[
            {"on": "yes", "next": "a"}])]},                          # bad on
        {"start": "a", "nodes": [dict(node, routes=[
            {"on": "unknown"}])]},                                   # no next
        {"start": "a", "nodes": [dict(node, routes=[
            {"on": "unknown", "next": "ghost"}])]},                  # unknown next
        {"start": "a", "nodes": [dict(node, routes=[
            {"on": "unknown", "conclusion": "x", "next": "a"}])]},   # conclusion
        {"start": "a", "nodes": [dict(node, routes=[
            {"on": "decided", "conclusion": "", "next": "a"}])]},    # empty concl.
        {"start": "a", "nodes": [dict(node, routes=[
            {"on": "decided", "conclusion": "x", "next": "a"},
            {"on": "decided", "conclusion": "x", "next": "a"}])]},   # dup route
        {"start": "a", "nodes": [dict(node, wow=1)]},                # node key
        {"start": "a", "nodes": [dict(node, routes=[
            {"on": "unknown", "next": "a", "wow": 1}])]},            # route key
        {"start": "a", "nodes": [node], "facts": []},                # bad facts
    ]
    for body in bad_bodies:
        r = post_flow(body)
        assert r.status_code == 422, body
        out = r.json()
        assert out["detail"] == "Flow validation failed"
        assert isinstance(out["errors"], list) and out["errors"]


def test_multiple_errors_are_reported_together():
    ids = create_policies()
    body = {
        "start": "ghost",
        "nodes": [{"id": "a", "policy_id": "nope",
                   "routes": [{"on": "unknown", "next": "ghost"}]}],
    }
    r = post_flow(body)
    assert r.status_code == 422
    errors = r.json()["errors"]
    assert any(e.startswith("start:") for e in errors)
    assert any("policy" in e and "not found" in e for e in errors)
    assert any("unknown node id" in e for e in errors)


def test_node_and_route_limits():
    ids = create_policies()
    many_nodes = [
        {"id": f"n{i}", "policy_id": ids["screen"]} for i in range(1001)
    ]
    r = post_flow({"start": "n0", "nodes": many_nodes})
    assert r.status_code == 422
    assert any("at most" in e for e in r.json()["errors"])

    many_routes = [
        {"on": "decided", "conclusion": f"c{i}", "next": "a"}
        for i in range(101)
    ]
    r = post_flow({"start": "a", "nodes": [
        {"id": "a", "policy_id": ids["screen"], "routes": many_routes}]})
    assert r.status_code == 422
    assert any("routes" in e and "at most" in e for e in r.json()["errors"])


def test_malformed_json_is_400():
    r = client.post(
        "/api/v1/adjudication-flows/evaluate",
        content="{not json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 400


# --------------------------------------------------------------------------- #
# Storage guarantees
# --------------------------------------------------------------------------- #

def test_flow_evaluation_never_writes_to_store():
    ids = create_policies()
    before = {
        pid: client.get(f"/api/v1/policies/{pid}").json()
        for pid in ids.values()
    }
    count_before = store.count()

    body = loan_flow(
        ids, [{"on": "decided", "conclusion": "allow", "next": "risk"}],
        FULL_FACTS,
    )
    assert post_flow(body).status_code == 200
    assert store.count() == count_before
    for pid, snapshot in before.items():
        assert client.get(f"/api/v1/policies/{pid}").json() == snapshot

    # A rejected (422) flow does not create or modify policies either.
    r = post_flow({"start": "a", "nodes": [{"id": "a", "policy_id": "nope"}]})
    assert r.status_code == 422
    assert store.count() == count_before
    for pid, snapshot in before.items():
        assert client.get(f"/api/v1/policies/{pid}").json() == snapshot


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for test in tests:
        setup_function()
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:
            failures += 1
            import traceback
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
