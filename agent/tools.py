from __future__ import annotations

from typing import Any, Dict, List, Optional
import pandas as pd

DEST_CSV = "data/destinations.csv"
ATTR_CSV = "data/attractions.csv"

def search_destinations(region: Optional[str], interests: List[str]) -> List[Dict[str, Any]]:
    df = pd.read_csv(DEST_CSV)
    if region:
        df = df[df["region"].str.lower() == region.lower()]

    # Score by tag overlap
    def score_tags(row) -> int:
        tags = str(row["style_tags"]).lower().split(",")
        tags = [t.strip() for t in tags]
        return sum(1 for i in interests if i.lower() in tags)

    df["score"] = df.apply(score_tags, axis=1)
    df = df.sort_values(["score", "avg_lodging_per_night"], ascending=[False, True])

    return df.head(4).to_dict(orient="records")

def propose_trip_concepts(constraints: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    interests = constraints.get("interests", [])
    pace = constraints.get("pace", "medium")

    concepts = []
    for c in candidates:
        city = c["city"]
        tags = str(c["style_tags"])
        concepts.append({
            "title": f"{city} — {pace} pace, aligned with {', '.join(interests) if interests else 'general sightseeing'}",
            "city": city,
            "why": f"Tags: {tags}. Good fit for budget and interests based on dataset.",
            "destination": c
        })
    return concepts

def build_itinerary(city: str, days: int, interests: List[str], pace: str) -> Dict[str, Any]:
    df = pd.read_csv(ATTR_CSV)
    df = df[df["city"].str.lower() == city.lower()].copy()

    # Prefer attractions matching interests; otherwise fallback to any
    if interests:
        df["match"] = df["tag"].str.lower().apply(lambda t: 1 if any(i.lower() in t for i in interests) else 0)
        df = df.sort_values(["match", "cost_est"], ascending=[False, True])
    else:
        df = df.sort_values(["cost_est"], ascending=[True])

    # Pace controls number of blocks/day
    blocks_per_day = {"slow": 2, "medium": 3, "fast": 4}.get(pace, 3)

    items = df.to_dict(orient="records")
    plan = []
    idx = 0
    for d in range(1, days + 1):
        day_items = []
        for _ in range(blocks_per_day):
            if idx >= len(items):
                break
            day_items.append(items[idx])
            idx += 1
        plan.append({"day": d, "items": day_items})

    return {"city": city, "days": days, "pace": pace, "plan": plan}

def estimate_budget(destination: Dict[str, Any], days: int, travelers: int, flight_est_per_person: float) -> Dict[str, Any]:
    lodging = float(destination["avg_lodging_per_night"]) * (days - 1)  # nights
    food = float(destination["avg_food_per_day"]) * days * travelers
    local = float(destination["avg_local_transport_per_day"]) * days * travelers
    flights = float(flight_est_per_person) * travelers

    total = lodging + food + local + flights
    return {
        "lodging": round(lodging, 2),
        "food": round(food, 2),
        "local_transport": round(local, 2),
        "flights_est": round(flights, 2),
        "total_est": round(total, 2),
    }

def estimate_activity_cost(itinerary: Dict[str, Any]) -> float:
    total = 0.0
    for day in itinerary["plan"]:
        for it in day["items"]:
            total += float(it["cost_est"])
    return round(total, 2)

def adjust_itinerary_for_budget(itinerary: Dict[str, Any], max_paid_activities: int) -> Dict[str, Any]:
    # Keep only first N paid items; free items remain.
    paid_seen = 0
    for day in itinerary["plan"]:
        new_items = []
        for it in day["items"]:
            cost = float(it["cost_est"])
            if cost <= 0:
                new_items.append(it)
                continue
            if paid_seen < max_paid_activities:
                new_items.append(it)
                paid_seen += 1
        day["items"] = new_items
    return itinerary

def build_checklist(constraints: Dict[str, Any], destination: Dict[str, Any]) -> List[str]:
    city = destination["city"]
    days = constraints["days"]
    return [
        f"Confirm travel dates and time off for {days} days",
        f"Book flights to {city} (compare 2–3 options)",
        "Book lodging near a transit-friendly area",
        "Save key attractions to a map list",
        "Set a daily spend cap and track expenses",
        "Prepare documents (passport/ID, cards, insurance as needed)",
        "Build a backup plan (weather / closures / fatigue day)",
    ]
