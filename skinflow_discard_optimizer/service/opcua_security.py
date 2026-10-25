"""OPC-UA security settings, from the environment, for both ends of the service.

Client side (talking to the press):
    SKDO_OPCUA_CERT, SKDO_OPCUA_KEY            our application certificate (DER or PEM) and private key (PEM)
    SKDO_OPCUA_SERVER_CERT                     the press server's certificate, pinned (PEM or DER)
    SKDO_OPCUA_USER, SKDO_OPCUA_PASSWORD       optional username login
    With cert + key + server cert set, the channel is Basic256Sha256 SignAndEncrypt; otherwise it is
    unsecured and a warning is logged.

Server side (the read-only Predictor.* server this service hosts):
    SKDO_SERVE_CERT, SKDO_SERVE_KEY            set both to require Basic256Sha256 SignAndEncrypt
    SKDO_SERVE_USER, SKDO_SERVE_PASSWORD       set both to require this username login (no anonymous access)
    SKDO_ALLOW_INSECURE=1                      allow an unsecured server on a non-loopback address

An unsecured Predictor server is only allowed on a loopback address unless SKDO_ALLOW_INSECURE=1.
`skdo gen-cert` creates a self-signed certificate and key.
"""
from __future__ import annotations

import hmac
import ipaddress
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from asyncua import ua
from asyncua.server.user_managers import User, UserManager, UserRole

logger = logging.getLogger("skinflow.opcua.security")


@dataclass(frozen=True)
class ServerSecurity:
    cert: Path | None = None
    key: Path | None = None
    user: str | None = None
    password: str | None = None
    allow_insecure: bool = False

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "ServerSecurity":
        env = dict(os.environ) if env is None else env
        cert, key = env.get("SKDO_SERVE_CERT"), env.get("SKDO_SERVE_KEY")
        if bool(cert) != bool(key):
            raise ValueError("SKDO_SERVE_CERT and SKDO_SERVE_KEY must be set together")
        user, pw = env.get("SKDO_SERVE_USER"), env.get("SKDO_SERVE_PASSWORD")
        if bool(user) != bool(pw):
            raise ValueError("SKDO_SERVE_USER and SKDO_SERVE_PASSWORD must be set together")
        return cls(Path(cert) if cert else None, Path(key) if key else None, user, pw,
                   env.get("SKDO_ALLOW_INSECURE") == "1")

    @property
    def encrypted(self) -> bool:
        return self.cert is not None and self.key is not None

    def check_bind(self, endpoint: str) -> None:
        """Refuse an unsecured server on a routable address."""
        if self.encrypted or self.allow_insecure:
            return
        if not is_loopback(urlparse(endpoint).hostname or ""):
            raise PermissionError(
                f"refusing to serve {endpoint} without encryption. Set SKDO_SERVE_CERT/SKDO_SERVE_KEY "
                "(see `skdo gen-cert`), bind to 127.0.0.1, or set SKDO_ALLOW_INSECURE=1 on an isolated network.")


def is_loopback(host: str) -> bool:
    if host in ("localhost", ""):
        return host == "localhost"
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class PasswordUserManager(UserManager):
    """Single configured user; anonymous and unknown users are rejected. Constant-time comparison."""

    def __init__(self, user: str, password: str):
        self._user, self._password = user.encode(), password.encode()

    def get_user(self, iserver, username=None, password=None, certificate=None):
        if username is None or password is None:
            return None
        ok_user = hmac.compare_digest(username.encode(), self._user)
        ok_pw = hmac.compare_digest(password.encode(), self._password)
        return User(role=UserRole.User, name=username) if (ok_user and ok_pw) else None


async def apply_server_security(server, sec: ServerSecurity) -> None:
    if sec.encrypted:
        await server.load_certificate(sec.cert)
        await server.load_private_key(sec.key)
        server.set_security_policy([ua.SecurityPolicyType.Basic256Sha256_SignAndEncrypt])
    else:
        logger.warning("Predictor server is NOT encrypted (loopback or SKDO_ALLOW_INSECURE)")
    if sec.user:
        server.set_identity_tokens([ua.UserNameIdentityToken])


def make_server(sec: ServerSecurity):
    """A ``Server`` whose user manager is fixed at construction (setting it afterwards has no effect)."""
    from asyncua import Server

    return Server(user_manager=PasswordUserManager(sec.user, sec.password or "") if sec.user else None)


async def apply_client_security(client, env: dict[str, str] | None = None) -> None:
    env = dict(os.environ) if env is None else env
    user, pw = env.get("SKDO_OPCUA_USER"), env.get("SKDO_OPCUA_PASSWORD")
    cert, key, server_cert = env.get("SKDO_OPCUA_CERT"), env.get("SKDO_OPCUA_KEY"), env.get("SKDO_OPCUA_SERVER_CERT")
    if cert or key or server_cert:
        if not (cert and key and server_cert):
            raise ValueError("SKDO_OPCUA_CERT, SKDO_OPCUA_KEY and SKDO_OPCUA_SERVER_CERT must all be set")
        await client.set_security_string(f"Basic256Sha256,SignAndEncrypt,{cert},{key},{server_cert}")
    else:
        logger.warning("OPC-UA client channel to the press is NOT encrypted")
    if user:
        client.set_user(user)
        client.set_password(pw or "")


async def generate_certificate(out_dir: Path, name: str, host: str, uri: str) -> tuple[Path, Path]:
    """Self-signed application certificate + RSA key (PEM) for OPC-UA; returns (cert, key)."""
    from asyncua.crypto.cert_gen import setup_self_signed_certificate
    from cryptography.x509.oid import ExtendedKeyUsageOID

    out_dir.mkdir(parents=True, exist_ok=True)
    cert, key = out_dir / f"{name}-cert.der", out_dir / f"{name}-key.pem"
    await setup_self_signed_certificate(
        key, cert, uri, host,
        [ExtendedKeyUsageOID.CLIENT_AUTH, ExtendedKeyUsageOID.SERVER_AUTH],
        {"countryName": "XX", "stateOrProvinceName": "NA", "localityName": "NA", "organizationName": "SKDO"})
    try:
        key.chmod(0o600)
    except OSError:
        pass                     # filesystems without POSIX permissions (FAT/exFAT on a USB stick)
    return cert, key
