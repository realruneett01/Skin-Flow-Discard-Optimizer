"""USB bundle: target parsing, manifest integrity, embedded Windows Python, preflight binding."""
import zipfile

import pytest

from skinflow_discard_optimizer.deploy import bundle, preflight


def test_parse_target():
    assert bundle.parse_target("cp312-win_amd64") == ("cp312", "3.12", "win_amd64")
    assert bundle.parse_target("cp313-linux_aarch64") == ("cp313", "3.13", "linux_aarch64")
    for bad in ("py312-win_amd64", "cp312-solaris", "cp312"):
        with pytest.raises(ValueError):
            bundle.parse_target(bad)


def test_runtime_dependencies_include_live_opcua():
    deps = " ".join(bundle.runtime_dependencies())
    assert "asyncua" in deps and "numpy" in deps


def _tree(root):
    (root / "app" / "config").mkdir(parents=True)
    (root / "app" / "config" / "press.yaml").write_text("a: 1\n")
    (root / "lib").mkdir()
    (root / "lib" / "x.py").write_text("x = 1\n")
    (root / "app" / "artifacts").mkdir()
    (root / "app" / "artifacts" / "state.json").write_text("{}")


def test_manifest_detects_tamper_missing_and_extra(tmp_path):
    _tree(tmp_path)
    bundle.write_manifest(tmp_path)
    assert bundle.verify_manifest(tmp_path) == []
    (tmp_path / "app" / "artifacts" / "state.json").write_text('{"changed": true}')   # mutable state is allowed to change
    (tmp_path / "app" / "__pycache__").mkdir()
    (tmp_path / "app" / "__pycache__" / "m.pyc").write_bytes(b"\0")                    # bytecode caches are ignored
    assert bundle.verify_manifest(tmp_path) == []
    (tmp_path / "app" / "config" / "press.yaml").write_text("a: 2\n")
    (tmp_path / "lib" / "evil.py").write_text("import os\n")
    (tmp_path / "lib" / "x.py").unlink()
    problems = " ".join(bundle.verify_manifest(tmp_path))
    assert "modified: app/config/press.yaml" in problems
    assert "unexpected file: lib/evil.py" in problems
    assert "missing: lib/x.py" in problems


def test_missing_manifest_is_a_failure(tmp_path):
    assert bundle.verify_manifest(tmp_path) == ["MANIFEST.sha256 is missing"]


def test_embedded_python_gets_search_paths(tmp_path):
    z = tmp_path / "py.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("python312._pth", "python312.zip\n.\n# import site\n")
        f.writestr("python.exe", "")
    out = tmp_path / "out"
    bundle._embed_python(z, out, ["cp312-win_amd64"])
    pth = (out / "runtime" / "win_amd64" / "python312._pth").read_text()
    assert "..\\..\\app" in pth and "..\\..\\lib\\cp312-win_amd64" in pth
    with pytest.raises(ValueError):
        bundle._embed_python(z, out, ["cp312-linux_x86_64"])


def test_launchers_written_and_unix_launcher_has_lf_endings(tmp_path):
    bundle._write_launchers(tmp_path, ["cp312-win_amd64"])
    assert b"\r" not in (tmp_path / "start.sh").read_bytes()
    assert b"\r\n" in (tmp_path / "START-WINDOWS.bat").read_bytes()
    assert "verify-bundle" in (tmp_path / "start.sh").read_text()
    assert "verify-bundle" in (tmp_path / "START-WINDOWS.bat").read_text()


def test_preflight_without_result_is_not_current(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "RESULT_FILE", tmp_path / "preflight.json")
    ok, why = preflight.is_current()
    assert not ok and "no preflight" in why
