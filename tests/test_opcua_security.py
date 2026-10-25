"""OPC-UA security: environment parsing, bind policy, and a real encrypted + authenticated round trip."""
import asyncio
import socket

import pytest
from asyncua import Client

from skinflow_discard_optimizer.contracts.schema import load_contract
from skinflow_discard_optimizer.service.opcua_security import (
    PasswordUserManager,
    ServerSecurity,
    generate_certificate,
    is_loopback,
)
from skinflow_discard_optimizer.service.opcua_source import PredictorServer


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("host,expected", [
    ("127.0.0.1", True), ("127.5.5.5", True), ("::1", True), ("localhost", True),
    ("0.0.0.0", False), ("192.168.1.5", False), ("10.0.0.1", False), ("press.example.com", False), ("", False),
])
def test_is_loopback(host, expected):
    assert is_loopback(host) is expected


def test_env_requires_pairs():
    with pytest.raises(ValueError):
        ServerSecurity.from_env({"SKDO_SERVE_CERT": "c.pem"})
    with pytest.raises(ValueError):
        ServerSecurity.from_env({"SKDO_SERVE_USER": "u"})
    s = ServerSecurity.from_env({"SKDO_SERVE_CERT": "c", "SKDO_SERVE_KEY": "k", "SKDO_SERVE_USER": "u",
                                 "SKDO_SERVE_PASSWORD": "p"})
    assert s.encrypted and s.user == "u"
    assert not ServerSecurity.from_env({}).encrypted


def test_unencrypted_routable_bind_refused_unless_overridden():
    plain = ServerSecurity()
    plain.check_bind("opc.tcp://127.0.0.1:4841/skdo")
    with pytest.raises(PermissionError):
        plain.check_bind("opc.tcp://0.0.0.0:4841/skdo")
    with pytest.raises(PermissionError):
        plain.check_bind("opc.tcp://192.168.0.5:4841/skdo")
    ServerSecurity(allow_insecure=True).check_bind("opc.tcp://0.0.0.0:4841/skdo")


def test_password_manager():
    m = PasswordUserManager("op", "s3cret")
    assert m.get_user(None, "op", "s3cret") is not None
    for u, p in [("op", "wrong"), ("other", "s3cret"), (None, None), ("op", None), ("", "")]:
        assert m.get_user(None, u, p) is None


def test_encrypted_authenticated_roundtrip_and_read_only(tmp_path):
    async def go():
        cert, key = await generate_certificate(tmp_path, "srv", "localhost", "urn:localhost:skdo:srv")
        ccert, ckey = await generate_certificate(tmp_path, "cli", "localhost", "urn:localhost:skdo:cli")
        ep = f"opc.tcp://127.0.0.1:{free_port()}/skdo"
        contract = load_contract()
        sec = ServerSecurity(cert, key, "op", "s3cret")
        server = PredictorServer(contract, ep, sec)
        await server.start()
        try:
            tag = contract.outputs.signals["Predictor.ButtCutMm"].tag
            await server.publish({"Predictor.ButtCutMm": 27.5})

            async def client(user, pw, secure=True):
                c = Client(ep, timeout=5)
                if secure:
                    await c.set_security_string(f"Basic256Sha256,SignAndEncrypt,{ccert},{ckey},{cert}")
                c.set_user(user)
                c.set_password(pw)
                return c

            c = await client("op", "s3cret")
            async with c:
                node = c.get_node(tag)
                assert await node.read_value() == pytest.approx(27.5)
                with pytest.raises(Exception):
                    await node.write_value(1.0)                      # read-only for clients
            with pytest.raises(Exception):
                async with await client("op", "wrong"):               # bad password
                    pass
            with pytest.raises(Exception):
                async with await client("op", "s3cret", secure=False):  # no encryption offered
                    pass
            anon = Client(ep, timeout=5)
            await anon.set_security_string(f"Basic256Sha256,SignAndEncrypt,{ccert},{ckey},{cert}")
            with pytest.raises(Exception):
                async with anon:                                      # anonymous login
                    pass
        finally:
            await server.stop()

    asyncio.run(asyncio.wait_for(go(), 60))
