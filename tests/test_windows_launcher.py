"""Test the Windows one-click launcher without opening browsers or model services."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from public_sector_opportunity_research.opportunity_explorer import (
    load_explorer_agencies,
)
from public_sector_opportunity_research.windows_launcher import (
    APP_ID,
    LauncherError,
    LauncherState,
    build_streamlit_command,
    launcher_paths,
    managed_instance_is_ready,
    resolve_project_root,
    start_explorer,
    stop_explorer,
    validate_installation,
    _spawn_supervisor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _FakeProcess:
    """Provide the process methods used by ``start_explorer`` unit tests."""

    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self.terminated = False

    def poll(self) -> int | None:
        """Stay alive until the launcher explicitly terminates this fake."""

        return 0 if self.terminated else None

    def terminate(self) -> None:
        """Record termination without touching a real process."""

        self.terminated = True


def _fake_project(tmp_path: Path, name: str = "Project With Spaces") -> Path:
    """Create only the local files the launcher validates before startup."""

    root = tmp_path / name
    for directory in (
        root / ".venv" / "Scripts",
        root / "src" / "public_sector_opportunity_research",
        root / "data",
        root / "outputs",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for file_path in (
        root / ".venv" / "Scripts" / "python.exe",
        root / ".venv" / "Scripts" / "pythonw.exe",
        root
        / "src"
        / "public_sector_opportunity_research"
        / "review_dashboard.py",
        root / "data" / "dol_research.sqlite3",
    ):
        file_path.write_bytes(b"test")
    return root


def _ready_start(root: Path, **overrides):
    """Start a fake managed instance while preserving real state-file behavior."""

    process = _FakeProcess()
    calls: list[tuple[int, int]] = []

    def spawn(paths, *, app_port, control_port, token):
        del paths, token
        calls.append((app_port, control_port))
        return process

    availability = overrides.pop("availability", lambda port: True)
    result = start_explorer(
        launcher_paths(root),
        open_browser=False,
        spawner=spawn,
        ready_waiter=lambda state, child: True,
        availability=availability,
        **overrides,
    )
    return result, process, calls


def test_project_root_resolves_outside_repository_and_handles_spaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Current working directory must not influence launcher path resolution."""

    root = tmp_path / "CMU Capstone" / "public-sector-opportunity-research"
    script = root / "src" / "public_sector_opportunity_research" / "windows_launcher.py"
    script.parent.mkdir(parents=True)
    script.write_text("", encoding="utf-8")
    elsewhere = tmp_path / "Elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert resolve_project_root(script_path=script) == root.resolve()
    assert resolve_project_root(root) == root.resolve()


def test_streamlit_command_uses_venv_loopback_and_existing_dashboard_arguments(
    tmp_path: Path,
) -> None:
    """Spaces must remain safe because startup uses an argument list, not a shell."""

    paths = launcher_paths(_fake_project(tmp_path))
    command = build_streamlit_command(paths, app_port=8507, token="marker-token")

    assert command[0] == str(paths.python)
    assert command[1:4] == ["-m", "streamlit", "run"]
    assert str(paths.dashboard) in command
    assert command[command.index("--server.address") + 1] == "127.0.0.1"
    assert command[command.index("--server.port") + 1] == "8507"
    assert command[command.index("--db") + 1] == str(paths.database)
    assert command[command.index("--outputs") + 1] == str(paths.outputs)
    assert command[command.index("--launcher-token") + 1] == "marker-token"


def test_start_when_not_running_writes_managed_state(tmp_path: Path) -> None:
    """A normal first launch starts once and records its authenticated controller."""

    root = _fake_project(tmp_path)
    result, _, calls = _ready_start(root)
    state = json.loads(launcher_paths(root).state_file.read_text(encoding="utf-8"))

    assert result.action == "started"
    assert result.port == 8501
    assert calls == [(8501, 8601)]
    assert state["app_id"] == APP_ID
    assert state["app_port"] == 8501
    assert state["token"]


def test_supervisor_inherits_user_scope_brave_key_without_command_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The desktop boundary passes a late-added key only in child environment."""

    paths = launcher_paths(_fake_project(tmp_path))
    captured: dict[str, object] = {}

    def popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return _FakeProcess()

    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    monkeypatch.setattr(
        "public_sector_opportunity_research.windows_launcher._brave_key_for_child",
        lambda: "configured-value",
    )
    monkeypatch.setattr(
        "public_sector_opportunity_research.windows_launcher.subprocess.Popen", popen
    )
    _spawn_supervisor(paths, app_port=8501, control_port=8601, token="control")

    assert captured["env"]["BRAVE_SEARCH_API_KEY"] == "configured-value"
    assert "configured-value" not in captured["command"]


def test_reopening_verified_instance_does_not_spawn_duplicate(tmp_path: Path) -> None:
    """A second launch must open the existing URL without creating a process."""

    root = _fake_project(tmp_path)
    _ready_start(root)
    opened: list[str] = []

    def unexpected_spawn(*args, **kwargs):
        del args, kwargs
        raise AssertionError("A verified instance must not spawn again")

    result = start_explorer(
        launcher_paths(root),
        opener=opened.append,
        instance_checker=lambda state: True,
        spawner=unexpected_spawn,
    )

    assert result.action == "reused"
    assert opened == ["http://127.0.0.1:8501"]


def test_unrelated_occupied_port_is_skipped_without_interference(tmp_path: Path) -> None:
    """Port 8501 occupancy alone must never identify or stop this application."""

    root = _fake_project(tmp_path)
    availability = lambda port: port not in {8501, 8601}
    result, process, calls = _ready_start(root, availability=availability)

    assert result.port == 8502
    assert calls == [(8502, 8602)]
    assert process.terminated is False


def test_stale_state_is_replaced_without_killing_saved_pid(tmp_path: Path) -> None:
    """Failed control identity makes state stale, not permission to kill a PID."""

    root = _fake_project(tmp_path)
    paths = launcher_paths(root)
    _ready_start(root)
    old_state = json.loads(paths.state_file.read_text(encoding="utf-8"))
    result, process, calls = _ready_start(
        root,
        instance_checker=lambda state: False,
    )
    new_state = json.loads(paths.state_file.read_text(encoding="utf-8"))

    assert result.action == "started"
    assert calls == [(8501, 8601)]
    assert process.terminated is False
    assert new_state["token"] != old_state["token"]


def test_safe_stop_uses_authenticated_controller_not_process_kill(tmp_path: Path) -> None:
    """Shutdown sends only the saved control port/token to the owned controller."""

    root = _fake_project(tmp_path)
    paths = launcher_paths(root)
    _ready_start(root)
    state = LauncherState.from_mapping(
        json.loads(paths.state_file.read_text(encoding="utf-8"))
    )
    requests: list[tuple[int, str, str]] = []

    def stop_request(port: int, token: str, action: str):
        requests.append((port, token, action))
        return {
            "app_id": APP_ID,
            "project_root": state.project_root,
            "app_port": state.app_port,
            "status": "stopped",
        }

    result = stop_explorer(paths, control_request=stop_request)

    assert result.action == "stopped"
    assert requests == [(state.control_port, state.token, "stop")]
    assert not paths.state_file.exists()


def test_stop_with_stale_record_clears_only_state(tmp_path: Path) -> None:
    """An unreachable control endpoint cannot authorize process termination."""

    root = _fake_project(tmp_path)
    paths = launcher_paths(root)
    _ready_start(root)

    result = stop_explorer(paths, control_request=lambda *args: None)

    assert result.action == "stale_record"
    assert not paths.state_file.exists()


def test_missing_environment_produces_beginner_friendly_error(tmp_path: Path) -> None:
    """A missing .venv must be reported before any process launch is attempted."""

    root = tmp_path / "Missing Project"
    root.mkdir()

    with pytest.raises(LauncherError, match="Python environment is missing"):
        validate_installation(launcher_paths(root), streamlit_checker=lambda: True)


def test_managed_identity_requires_control_token_and_http_health() -> None:
    """A generic server on the app port cannot be mistaken for this launcher."""

    state = LauncherState(
        app_id=APP_ID,
        project_root=r"C:\Example Project",
        app_port=8501,
        control_port=8601,
        token="private-random-marker",
        supervisor_pid=42,
        started_at="2026-11-20T00:00:00+00:00",
    )
    response = {
        "app_id": APP_ID,
        "project_root": state.project_root,
        "app_port": 8501,
        "status": "running",
    }

    assert managed_instance_is_ready(
        state,
        control_request=lambda *args: response,
        health_check=lambda port: True,
    )
    assert not managed_instance_is_ready(
        state,
        control_request=lambda *args: {**response, "app_id": "other-app"},
        health_check=lambda port: True,
    )
    assert not managed_instance_is_ready(
        state,
        control_request=lambda *args: response,
        health_check=lambda port: False,
    )


def test_existing_local_data_is_available_without_network() -> None:
    """Launcher prerequisites and all five Explorer agencies load from disk only."""

    paths = launcher_paths(PROJECT_ROOT)
    validate_installation(paths)
    agencies = load_explorer_agencies(PROJECT_ROOT)

    assert [agency.registry.abbreviation for agency in agencies] == [
        "DOL",
        "VA",
        "HHS",
        "EDD",
        "IDES",
    ]
    assert all(agency.available for agency in agencies)


def test_shortcut_scripts_keep_setup_separate_and_use_hidden_wscript() -> None:
    """Normal startup must not activate PowerShell or install system software."""

    start_vbs = (PROJECT_ROOT / "windows" / "Launch Opportunity Explorer.vbs").read_text(
        encoding="utf-8"
    )
    stop_vbs = (PROJECT_ROOT / "windows" / "Stop Opportunity Explorer.vbs").read_text(
        encoding="utf-8"
    )
    installer = (PROJECT_ROOT / "windows" / "Install Desktop Shortcuts.ps1").read_text(
        encoding="utf-8"
    )

    assert ".venv\\Scripts\\pythonw.exe" in start_vbs
    assert "shell.Run command, 0, False" in start_vbs
    assert " start --project-root " in start_vbs
    assert " stop --project-root " in stop_vbs
    assert "Public Sector Opportunity Explorer" in installer
    assert "Stop Opportunity Explorer" in installer
    assert "WScript.Shell" in installer
