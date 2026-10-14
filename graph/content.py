"""``content_id``: identity of *what was measured*, without *which run recorded it*.

``object_id`` hashes the payload, which holds the causal-log blob, whose records carry
``trajectory_hash`` (a hash that includes ``experiment_id``/``run_id``). Two runs of the same
measurement under different experiment ids therefore have different ``object_id`` (and that is kept:
the experiment id is part of the record). ``content_id`` is the hash of the same records with the
run-identifying fields removed, plus the ``space_id``. It is stored beside ``object_id``, never
instead of it.
"""

from __future__ import annotations

import hashlib
from typing import Any

from graph.chainlog import canonical

#: record-level fields that identify the run, not the measured content
RUN_FIELDS = ("trajectory_hash",)
#: observation-level extras added by the kernel (calibration conditions), outside ``result``
OBSERVATION_EXTRAS = ("annotations",)


def _strip_ports(ports: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for port, obs in ports.items():
        out[port] = {k: v for k, v in obs.items() if k not in OBSERVATION_EXTRAS} if isinstance(obs, dict) else obs
    return out


def content_id(space_id: str, records: list[dict[str, Any]]) -> str:
    body = []
    for record in records:
        item = {k: v for k, v in record.items() if k not in RUN_FIELDS}
        item["ports"] = _strip_ports(item.get("ports", {}))
        body.append(item)
    payload = canonical({"schema": "hekbv3.content/1", "space_id": space_id, "records": body})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
