"""Shared strings for pre-line heuristic tests (simulated tool return shape)."""


def _make_tool_result(ask_next: str) -> str:
    """Simulate update_application_state tool return with Ask next."""
    return f'Updated. Still needed: foo. Ask next: "{ask_next}"'


OCCUPANCY_ASK_NEXT = (
    "What is the property type: primary, secondary, or investment?"
)
FOR_SALE_ASK_NEXT = "Is the property currently for sale?"
