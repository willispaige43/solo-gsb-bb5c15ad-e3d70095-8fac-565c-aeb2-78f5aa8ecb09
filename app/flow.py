"""Adjudication flows: chained policy evaluation with verdict-based routing.

A flow evaluation walks a directed graph of inline nodes, each referencing a
stored policy.  Every visited node adjudicates the same submitted ``facts``
with the regular engine; afterwards the node's routes are scanned in
declaration order and the first route matching the verdict — ``decided``
(optionally restricted to a specific conclusion), ``unknown``, ``conflict``
or ``no_match`` — determines the next node.  The walk ends at the first node
whose verdict matches no route.

Requests are fully validated before execution (see
``validation.validate_flow_request``): node ids, routes and policy references
are checked and the subgraph reachable from ``start`` must be acyclic, so the
walk below always terminates.  The policy store is only read, never written.
"""
from __future__ import annotations

from typing import Optional

from .engine import adjudicate
from .preview import verdict_signature

TERMINATION_NO_MATCHING_ROUTE = "no_matching_route"


def _match_route(routes: list[dict], verdict: dict) -> Optional[dict]:
    """First route matching ``verdict``; ``None`` when no route matches.

    A route matches when its ``on`` equals the verdict status and — for
    ``decided`` routes carrying a ``conclusion`` — the verdict conclusion
    equals it.  A ``decided`` route without ``conclusion`` matches any
    decided verdict.
    """
    status = verdict["status"]
    for route in routes:
        if route["on"] != status:
            continue
        if status == "decided":
            wanted = route.get("conclusion")
            if wanted is not None and wanted != verdict.get("conclusion"):
                continue
        return route
    return None


def evaluate_flow(
    start: str,
    nodes: list[dict],
    facts: dict,
    policy_by_id: dict[str, dict],
) -> dict:
    """Walk the validated flow from ``start`` and return the flow verdict.

    Returns ``termination_reason`` (why the walk ended), ``final_signature``
    (the business signature of the last verdict) and ``trace`` (one entry per
    visited node, in execution order, with the matched route or ``None`` on
    the terminal node).
    """
    node_by_id = {node["id"]: node for node in nodes}
    trace: list[dict] = []
    current_id = start
    while True:
        node = node_by_id[current_id]
        verdict = adjudicate(policy_by_id[node["policy_id"]], facts)
        signature = verdict_signature(verdict)
        route = _match_route(node["routes"], verdict)
        trace.append(
            {
                "node_id": node["id"],
                "policy_id": node["policy_id"],
                "status": verdict["status"],
                "conclusion": verdict.get("conclusion"),
                "signature": signature,
                "matched_route": route,
            }
        )
        if route is None:
            return {
                "termination_reason": TERMINATION_NO_MATCHING_ROUTE,
                "final_signature": signature,
                "trace": trace,
            }
        current_id = route["next"]
