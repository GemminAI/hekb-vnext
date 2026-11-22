"""The GateVerdict (gate-verdict/1) schema of sensos, vendored (D-91: no import of another part; fixed by LOCK.json).

``gate-verdict.schema.json`` is a byte-for-byte copy of sensos's file. ``validate.py`` is NOT sensos's: hekb-vnext's
environment has no ``jsonschema``, so a small validator of the keywords this schema uses is kept here (standard library only);
it is checked against ``jsonschema`` (draft 2020-12) in tests/test_gate_verdict_vendor.py (opt-in, needs GATE_PY).
"""
