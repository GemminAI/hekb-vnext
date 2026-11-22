"""A standard-library validator for the JSON Schema keywords of ``gate-verdict.schema.json`` (draft 2020-12).

Implemented keywords (all the schema uses): $ref (local ``#/$defs/...``), type, const, enum, required, properties,
additionalProperties, patternProperties, items, minItems, maxItems, uniqueItems, minLength, minimum, maximum, pattern,
allOf, oneOf, if / then / else. Annotation keywords ($schema, $id, $defs, title, description) are ignored.
Any other keyword in the schema is an error at load time, so a changed schema cannot be read silently with fewer rules.
JSON types are strict: ``True`` is not an integer, ``1`` and ``1.0`` are told apart only as Python does (the verdict has no floats).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).with_name("gate-verdict.schema.json")
_ANNOTATIONS = {"$schema", "$id", "$defs", "title", "description"}
_KEYWORDS = {
    "$ref",
    "type",
    "const",
    "enum",
    "required",
    "properties",
    "additionalProperties",
    "patternProperties",
    "items",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minLength",
    "minimum",
    "maximum",
    "pattern",
    "allOf",
    "oneOf",
    "if",
    "then",
    "else",
}


def _same(a: Any, b: Any) -> bool:
    """JSON equality: type-strict for bool / number, deep for containers."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return type(a) is type(b) and a == b


def _is(value: Any, name: str) -> bool:
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "null":
        return value is None
    raise ValueError(f"unknown type {name!r}")


def _check_keywords(node: Any) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("properties", "patternProperties", "$defs"):
                for sub in value.values():
                    _check_keywords(sub)
            elif key in _ANNOTATIONS:
                continue
            elif key not in _KEYWORDS:
                raise ValueError(
                    f"the schema uses a keyword this validator does not implement: {key}"
                )
            elif key in ("allOf", "oneOf"):
                for sub in value:
                    _check_keywords(sub)
            elif key in (
                "items",
                "additionalProperties",
                "if",
                "then",
                "else",
            ) and isinstance(value, dict):
                _check_keywords(value)


class Validator:
    def __init__(self, schema: Mapping[str, Any]) -> None:
        self.root = schema
        _check_keywords(schema)

    def errors(self, value: Any) -> list[str]:
        out: list[str] = []
        self._walk(self.root, value, "$", out)
        return out

    def _ref(self, ref: str) -> Any:
        if not ref.startswith("#/"):
            raise ValueError(f"only local references are supported: {ref}")
        node: Any = self.root
        for part in ref[2:].split("/"):
            node = node[part]
        return node

    def valid(self, schema: Any, value: Any) -> bool:
        out: list[str] = []
        self._walk(schema, value, "$", out)
        return not out

    def _walk(self, schema: Any, value: Any, path: str, out: list[str]) -> None:
        if schema is True or schema == {}:
            return
        if schema is False:
            out.append(f"{path}: not allowed")
            return
        if "$ref" in schema:
            self._walk(self._ref(schema["$ref"]), value, path, out)
        if "type" in schema:
            names = (
                [schema["type"]]
                if isinstance(schema["type"], str)
                else list(schema["type"])
            )
            if not any(_is(value, n) for n in names):
                out.append(f"{path}: not of type {names}")
                return
        if "const" in schema and not _same(value, schema["const"]):
            out.append(f"{path}: must be {schema['const']!r}")
        if "enum" in schema and not any(_same(value, e) for e in schema["enum"]):
            out.append(f"{path}: not one of {schema['enum']!r}")
        if isinstance(value, dict):
            for name in schema.get("required", ()):
                if name not in value:
                    out.append(f"{path}: '{name}' is required")
            props = schema.get("properties", {})
            patterns = schema.get("patternProperties", {})
            for key, item in value.items():
                matched = False
                if key in props:
                    matched = True
                    self._walk(props[key], item, f"{path}.{key}", out)
                for pattern, sub in patterns.items():
                    if re.search(pattern, key):
                        matched = True
                        self._walk(sub, item, f"{path}.{key}", out)
                if not matched and "additionalProperties" in schema:
                    extra = schema["additionalProperties"]
                    if extra is False:
                        out.append(f"{path}: unexpected key '{key}'")
                    elif extra is not True:
                        self._walk(extra, item, f"{path}.{key}", out)
        if isinstance(value, list):
            if "minItems" in schema and len(value) < schema["minItems"]:
                out.append(f"{path}: fewer than {schema['minItems']} items")
            if "maxItems" in schema and len(value) > schema["maxItems"]:
                out.append(f"{path}: more than {schema['maxItems']} items")
            if schema.get("uniqueItems"):
                for i, a in enumerate(value):
                    if any(_same(a, b) for b in value[:i]):
                        out.append(f"{path}: items are not unique")
                        break
            if "items" in schema:
                for i, item in enumerate(value):
                    self._walk(schema["items"], item, f"{path}[{i}]", out)
        if isinstance(value, str):
            if "minLength" in schema and len(value) < schema["minLength"]:
                out.append(f"{path}: shorter than {schema['minLength']}")
            if "pattern" in schema and not re.search(schema["pattern"], value):
                out.append(f"{path}: does not match {schema['pattern']!r}")
        if isinstance(value, int | float) and not isinstance(value, bool):
            if "minimum" in schema and value < schema["minimum"]:
                out.append(f"{path}: below {schema['minimum']}")
            if "maximum" in schema and value > schema["maximum"]:
                out.append(f"{path}: above {schema['maximum']}")
        for sub in schema.get("allOf", ()):
            self._walk(sub, value, path, out)
        if (
            "oneOf" in schema
            and sum(1 for sub in schema["oneOf"] if self.valid(sub, value)) != 1
        ):
            out.append(f"{path}: must match exactly one of the alternatives")
        if "if" in schema:
            branch = "then" if self.valid(schema["if"], value) else "else"
            if branch in schema:
                self._walk(schema[branch], value, path, out)


_VALIDATOR: Validator | None = None


def verdict_errors(verdict: Any) -> list[str]:
    """The reasons ``verdict`` does not fit gate-verdict/1 (empty = it fits)."""
    global _VALIDATOR
    if _VALIDATOR is None:
        _VALIDATOR = Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    return _VALIDATOR.errors(verdict)


def schema_sha256() -> str:
    import hashlib

    return hashlib.sha256(SCHEMA_PATH.read_bytes()).hexdigest()


__all__ = ["SCHEMA_PATH", "Validator", "schema_sha256", "verdict_errors"]
