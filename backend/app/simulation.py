"""Parser for the `X-Simulate-Failure` test hook.

Format: comma-separated `<key_id>=<status>` pairs, `*` meaning every key.
Allowed statuses: 429, 500, 502, 503.

    key_1=429            -> {"key_1": 429}
    key_1=429,key_2=500  -> {"key_1": 429, "key_2": 500}
    *=429                -> {"*": 429}
"""

from __future__ import annotations

import re

ALLOWED_STATUSES = frozenset({429, 500, 502, 503})
WILDCARD = "*"
_KEY_ID = re.compile(r"^(\*|key_[1-9][0-9]*)$")


class InvalidSimulationHeader(ValueError):
    pass


def parse_simulation_header(raw: str | None) -> dict[str, int]:
    if raw is None or not raw.strip():
        return {}
    plan: dict[str, int] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise InvalidSimulationHeader(f"expected <key_id>=<status>, got '{part}'")
        key_id, _, status_text = (piece.strip() for piece in part.partition("="))
        key_id = key_id.lower()
        if not _KEY_ID.match(key_id):
            raise InvalidSimulationHeader(f"invalid key id '{key_id}' (use key_1, key_2, ... or *)")
        try:
            status = int(status_text)
        except ValueError:
            raise InvalidSimulationHeader(f"invalid status '{status_text}'") from None
        if status not in ALLOWED_STATUSES:
            raise InvalidSimulationHeader(
                f"unsupported status {status} (allowed: {', '.join(map(str, sorted(ALLOWED_STATUSES)))})"
            )
        plan[key_id] = status
    return plan


def simulated_status_for(plan: dict[str, int], key_id: str) -> int | None:
    """The status a key must fake in this request, if any (explicit key wins over `*`)."""
    if key_id in plan:
        return plan[key_id]
    return plan.get(WILDCARD)
