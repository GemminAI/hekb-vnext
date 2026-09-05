"""Dev/test-only P-256 keypair + signer for `POST /experience`'s
`X-Audit-Signature` header (SPEC-HEKB-002_v2 §3.1).

This is NOT the real Secure Enclave key. The production key is generated
inside hardware by `MLBAKit/Security/SecureEnclaveManager.swift` and never
leaves it; `store_service.py` verifies against whatever public key
`HEKB_AUDIT_PUBLIC_KEY_PATH` points to, without caring how it got there.
This script exists only so that gate can be exercised end to end (a real
201, a real signature-rejected 403) before that hardware-backed export
pipeline exists.

Usage:
    python3 tools/dev_audit_signer.py keygen <out_dir>
        Writes <out_dir>/dev_private_key.pem (PKCS8, unencrypted — dev use
        only, never commit this) and <out_dir>/dev_public_key.raw (raw
        65-byte X9.63 uncompressed point — the same format
        `SecKeyCopyExternalRepresentation` produces for a P-256 key, so
        this is a drop-in stand-in for `HEKB_AUDIT_PUBLIC_KEY_PATH`).

    python3 tools/dev_audit_signer.py sign <private_key.pem> <payload.json>
        Prints a base64 `X-Audit-Signature` value for that payload file's
        canonical bytes — the exact JSON object that would go in a
        `POST /experience` body's `"payload"` field (reserved keys only,
        e.g. `{"schema": "...", "raw_input": "...", ...}`).
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vendor" / "hekb_vnext"))
from core.object import canonical_bytes  # noqa: E402


def keygen(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    private_key = ec.generate_private_key(ec.SECP256R1())

    private_path = out_dir / "dev_private_key.pem"
    private_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )

    numbers = private_key.public_key().public_numbers()
    raw_point = b"\x04" + numbers.x.to_bytes(32, "big") + numbers.y.to_bytes(32, "big")
    public_path = out_dir / "dev_public_key.raw"
    public_path.write_bytes(raw_point)

    print(f"private key: {private_path}")
    print(f"public key:  {public_path}  (point to this with HEKB_AUDIT_PUBLIC_KEY_PATH)")


def sign(private_key_path: Path, payload_path: Path) -> None:
    private_key = serialization.load_pem_private_key(private_key_path.read_bytes(), password=None)
    payload = json.loads(payload_path.read_text())
    message = canonical_bytes(payload)
    signature = private_key.sign(message, ec.ECDSA(hashes.SHA256()))
    print(base64.b64encode(signature).decode("ascii"))


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    command = sys.argv[1]
    if command == "keygen" and len(sys.argv) == 3:
        keygen(Path(sys.argv[2]))
    elif command == "sign" and len(sys.argv) == 4:
        sign(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        print(__doc__)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
