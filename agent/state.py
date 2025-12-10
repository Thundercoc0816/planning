from typing import Any, Dict, List, Optional, TypedDict

class AgentState(TypedDict, total=False):
    user_request: str

    constraints: Dict[str, Any]
    trip_options: List[Dict[str, Any]]

    selected_destination: Optional[Dict[str, Any]]
    itinerary: Optional[Dict[str, Any]]
    budget: Optional[Dict[str, Any]]
    checklist: Optional[List[str]]

    final_plan: str
    history: List[Dict[str, Any]]
