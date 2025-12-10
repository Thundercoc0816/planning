from __future__ import annotations

import re
from typing import Any, Dict, List

from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.tools import (
    search_destinations,
    propose_trip_concepts,
    build_itinerary,
    estimate_budget,
    estimate_activity_cost,
    adjust_itinerary_for_budget,
    build_checklist,
)

def _push(state: AgentState, node: str, payload: Dict[str, Any]) -> AgentState:
    h = state.get("history", [])
    h.append({"node": node, **payload})
    state["history"] = h
    return state

def _safe_int(x, default):
    try:
        return int(x)
    except Exception:
        return default

def parse_request(state: AgentState) -> AgentState:
    text = (state.get("user_request") or "").strip()

    # Defaults (demo-friendly)
    constraints: Dict[str, Any] = {
        "origin": "Boston",
        "days": 5,
        "travelers": 2,
        "budget": 2000,
        "region": None,  # europe / asia / americas
        "pace": "medium",  # slow/medium/fast
        "interests": ["food", "museums"],
        "month_hint": None,
        "flight_est_per_person": 450,  # placeholder
    }

    # days
    m = re.search(r"\b(\d{1,2})\s*-\s*day\b|\b(\d{1,2})\s*day\b", text.lower())
    if m:
        val = m.group(1) or m.group(2)
        constraints["days"] = _safe_int(val, constraints["days"])

    # budget
    m = re.search(r"\bbudget\s*\$?\s*(\d{3,6})\b|\bunder\s*\$?\s*(\d{3,6})\b", text.lower())
    if m:
        val = m.group(1) or m.group(2)
        constraints["budget"] = _safe_int(val, constraints["budget"])

    # travelers
    m = re.search(r"\bfor\s+(\d{1,2})\s+(people|persons|travelers)\b", text.lower())
    if m:
        constraints["travelers"] = _safe_int(m.group(1), constraints["travelers"])

    # region
    for r in ["europe", "asia", "americas"]:
        if r in text.lower():
            constraints["region"] = r
            break

    # pace
    for p in ["slow", "medium", "fast"]:
        if p in text.lower():
            constraints["pace"] = p
            break

    # interests (simple keyword scan)
    possible = ["food", "museums", "nature", "walk", "history", "shopping", "architecture", "coastal", "scenic", "art", "nightlife", "markets", "roadtrip"]
    found = [k for k in possible if k in text.lower()]
    if found:
        # Keep top 3 to avoid overfitting
        constraints["interests"] = found[:3]

    # month hint (optional)
    for mo in ["january","february","march","april","may","june","july","august","september","october","november","december"]:
        if mo in text.lower():
            constraints["month_hint"] = mo.title()
            break

    state["constraints"] = constraints
    return _push(state, "parse_request", {"constraints": constraints})

def make_options(state: AgentState) -> AgentState:
    c = state["constraints"]
    candidates = search_destinations(c.get("region"), c.get("interests", []))
    options = propose_trip_concepts(c, candidates)
    state["trip_options"] = options
    return _push(state, "propose_options", {"options": options})

def choose_destination(state: AgentState) -> AgentState:
    c = state["constraints"]
    options = state.get("trip_options", [])

    # Heuristic selection: lowest estimated total (rough)
    best = None
    best_total = float("inf")

    for opt in options:
        dest = opt["destination"]
        rough = estimate_budget(dest, c["days"], c["travelers"], c["flight_est_per_person"])
        total = rough["total_est"]
        if total < best_total:
            best_total = total
            best = dest

    state["selected_destination"] = best
    return _push(state, "choose_destination", {"selected_destination": best, "rough_total": best_total})

def build_plan(state: AgentState) -> AgentState:
    c = state["constraints"]
    dest = state["selected_destination"]
    itinerary = build_itinerary(dest["city"], c["days"], c["interests"], c["pace"])
    state["itinerary"] = itinerary
    return _push(state, "build_itinerary", {"itinerary_preview": itinerary["plan"][:1]})

def compute_budget(state: AgentState) -> AgentState:
    c = state["constraints"]
    dest = state["selected_destination"]
    base = estimate_budget(dest, c["days"], c["travelers"], c["flight_est_per_person"])
    act = estimate_activity_cost(state["itinerary"])
    total = base["total_est"] + act

    budget = {**base, "activities_est": act, "grand_total_est": round(total, 2), "within_budget": total <= c["budget"]}
    state["budget"] = budget
    return _push(state, "estimate_budget", {"budget": budget})

def validate_and_adjust(state: AgentState) -> AgentState:
    c = state["constraints"]
    budget = state["budget"]
    if budget["within_budget"]:
        return _push(state, "validate_adjust", {"action": "no_change"})

    # If over budget: reduce paid activities first
    over = budget["grand_total_est"] - c["budget"]

    # Rule: allow at most N paid activities depending on overage
    if over > 500:
        max_paid = 1
    elif over > 250:
        max_paid = 2
    else:
        max_paid = 3

    itinerary = adjust_itinerary_for_budget(state["itinerary"], max_paid_activities=max_paid)
    state["itinerary"] = itinerary

    # recompute budget
    base = {k: budget[k] for k in ["lodging","food","local_transport","flights_est","total_est"]}
    act = estimate_activity_cost(itinerary)
    grand = base["total_est"] + act
    new_budget = {**base, "activities_est": act, "grand_total_est": round(grand, 2), "within_budget": grand <= c["budget"]}
    state["budget"] = new_budget

    return _push(state, "validate_adjust", {"action": "reduced_paid_activities", "max_paid": max_paid, "new_budget": new_budget})

def finalize(state: AgentState) -> AgentState:
    c = state["constraints"]
    dest = state["selected_destination"]
    budget = state["budget"]
    itinerary = state["itinerary"]
    checklist = build_checklist(c, dest)
    state["checklist"] = checklist

    # Format itinerary
    lines: List[str] = []
    for day in itinerary["plan"]:
        lines.append(f"**Day {day['day']}**")
        if not day["items"]:
            lines.append("- Free exploration / rest day")
        else:
            for it in day["items"]:
                lines.append(f"- {it['name']} ({it['tag']}, ~{it['typical_hours']}h, est ${it['cost_est']})")
        lines.append("")

    interest_text = ", ".join(c.get("interests", [])) if c.get("interests") else "general"
    region_text = c.get("region") or "any"

    plan = (
        f"## Vacation Plan\n"
        f"**Destination:** {dest['city']}, {dest['country']}  \n"
        f"**Region preference:** {region_text}  \n"
        f"**Length:** {c['days']} days | **Travelers:** {c['travelers']}  \n"
        f"**Pace:** {c['pace']} | **Interests:** {interest_text}  \n\n"
        f"### Budget (estimates)\n"
        f"- Flights: ${budget['flights_est']}\n"
        f"- Lodging: ${budget['lodging']}\n"
        f"- Food: ${budget['food']}\n"
        f"- Local transport: ${budget['local_transport']}\n"
        f"- Activities: ${budget['activities_est']}\n"
        f"- **Grand total:** ${budget['grand_total_est']} (Budget: ${c['budget']})  \n"
        f"- **Within budget:** {budget['within_budget']}\n\n"
        f"### Day-by-day itinerary\n"
        + "\n".join(lines)
        + "\n### Booking & prep checklist\n"
        + "\n".join([f"- {x}" for x in checklist])
        + "\n"
    )

    state["final_plan"] = plan
    return _push(state, "finalize", {"final_plan_preview": plan[:300] + "..."})

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("parse_request", parse_request)
    g.add_node("propose_options", make_options)
    g.add_node("choose_destination", choose_destination)
    g.add_node("build_itinerary", build_plan)
    g.add_node("estimate_budget", compute_budget)
    g.add_node("validate_adjust", validate_and_adjust)
    g.add_node("finalize", finalize)

    g.set_entry_point("parse_request")
    g.add_edge("parse_request", "propose_options")
    g.add_edge("propose_options", "choose_destination")
    g.add_edge("choose_destination", "build_itinerary")
    g.add_edge("build_itinerary", "estimate_budget")
    g.add_edge("estimate_budget", "validate_adjust")
    g.add_edge("validate_adjust", "finalize")
    g.add_edge("finalize", END)

    return g.compile()
