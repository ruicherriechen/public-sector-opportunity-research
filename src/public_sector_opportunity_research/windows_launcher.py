"""Start and stop the local Opportunity Explorer safely on Windows.

This module sits in front of the existing Streamlit entry point.  The normal
double-click scripts run it with the project's ``pythonw.exe``, so no virtual
environment activation or persistent console window is needed.  It validates
the local installation, selects an available loopback port, starts Streamlit,
opens the default browser, and records only minimal local process metadata.

An authenticated control socket identifies launcher-owned instances.  A port
is never reused merely because it answers HTTP: the saved random token must
also receive the expected application identity from the controller process.
The stop command sends that token to the controller, which terminates only the
exact Streamlit child it created.  Research, retrieval, models, credentials,
and saved evidence are never touched by this module.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
from typing import Any, Callable, Iterator, Protocol, Sequence
from urllib.request import ProxyHandler, build_opener
import webbrowser


APP_ID = "public-sector-opportunity-explorer"
APP_TITLE = "Public Sector Opportunity Explorer"
APP_PORTS = range(8501, 8511)
CONTROL_PORT_OFFSET = 100
STATE_RELATIVE_PATH = Path(".tmp/opportunity_explorer_launcher.json")
LOCK_RELATIVE_PATH = Path(".tmp/opportunity_explorer_launcher.lock")
DEFAULT_STARTUP_TIMEOUT_SECONDS = 30.0


def _brave_key_for_child() -> str | None:
    """Resolve Brave configuration without printing or persisting its value."""

    process_value = (os.environ.get("BRAVE_SEARCH_API_KEY") or "").strip()
    if process_value:
        return process_value
    if os.name != "nt":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, "BRAVE_SEARCH_API_KEY")
    except (ImportError, FileNotFoundError, OSError):
        return None
    cleaned = str(value).strip() if value is not None else ""
    return cleaned or None


class LauncherError(RuntimeError):
    """Describe a startup problem in language suitable for a local user."""


class SpawnedProcess(Protocol):
    """Describe the small subprocess surface used by the controller."""

    pid: int

    def poll(self) -> int | None:
        """Return ``None`` while the process is running."""

    def terminate(self) -> None:
        """Request termination of this exact process."""


@dataclass(frozen=True)
class LauncherPaths:
    """Hold absolute paths needed to launch the existing dashboard."""

    project_root: Path
    python: Path
    pythonw: Path
    dashboard: Path
    database: Path
    outputs: Path
    data_directory: Path
    launcher_script: Path
    state_file: Path
    lock_file: Path


@dataclass(frozen=True)
class LauncherState:
    """Persist the minimum information needed to identify one managed server."""

    app_id: str
    project_root: str
    app_port: int
    control_port: int
    token: str
    supervisor_pid: int
    started_at: str

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "LauncherState":
        """Validate the shape and basic identity of a state-file mapping."""

        state = cls(
            app_id=str(value["app_id"]),
            project_root=str(value["project_root"]),
            app_port=int(value["app_port"]),
            control_port=int(value["control_port"]),
            token=str(value["token"]),
            supervisor_pid=int(value["supervisor_pid"]),
            started_at=str(value["started_at"]),
        )
        if state.app_id != APP_ID or not state.token:
            raise ValueError("State file does not identify this application.")
        return state


@dataclass(frozen=True)
class LaunchResult:
    """Report whether the launcher reused, started, or stopped an instance."""

    action: str
    url: str | None
    port: int | None


def resolve_project_root(
    explicit_root: Path | None = None, *, script_path: Path | None = None
) -> Path:
    """Resolve the repository independently of the caller's working directory.

    Args:
        explicit_root: Root supplied by the VBScript wrapper or a test.
        script_path: Optional launcher-file path used by tests. The real module
            defaults to ``__file__`` and lives under ``src/<package>``.

    Returns:
        An absolute repository path. The directory does not need to exist yet;
        :func:`validate_installation` provides the user-facing error.
    """

    if explicit_root is not None:
        return explicit_root.expanduser().resolve()
    module_path = (script_path or Path(__file__)).resolve()
    return module_path.parents[2]


def launcher_paths(project_root: Path) -> LauncherPaths:
    """Build every required path from one resolved repository root."""

    root = project_root.resolve()
    return LauncherPaths(
        project_root=root,
        python=root / ".venv" / "Scripts" / "python.exe",
        pythonw=root / ".venv" / "Scripts" / "pythonw.exe",
        dashboard=root
        / "src"
        / "public_sector_opportunity_research"
        / "review_dashboard.py",
        database=root / "data" / "dol_research.sqlite3",
        outputs=root / "outputs",
        data_directory=root / "data",
        launcher_script=Path(__file__).resolve(),
        state_file=root / STATE_RELATIVE_PATH,
        lock_file=root / LOCK_RELATIVE_PATH,
    )


def _streamlit_importable() -> bool:
    """Return whether Streamlit is installed in the current virtual environment."""

    return importlib.util.find_spec("streamlit") is not None


def validate_installation(
    paths: LauncherPaths,
    *,
    streamlit_checker: Callable[[], bool] = _streamlit_importable,
) -> None:
    """Raise one clear error when a required local component is unavailable.

    This check reads paths and package metadata only. It never initializes a
    database, loads a model, accesses credentials, or makes a network request.
    """

    checks: list[tuple[bool, str]] = [
        (
            paths.project_root.is_dir(),
            "The project folder could not be found. Move the shortcut only; "
            "do not move its project folder, then reinstall the shortcuts.",
        ),
        (
            paths.python.is_file() and paths.pythonw.is_file(),
            "The project's Python environment is missing. Recreate .venv and "
            "install the requirements before launching the application.",
        ),
        (
            paths.dashboard.is_file(),
            "The Opportunity Explorer dashboard file is missing from the project.",
        ),
        (
            paths.data_directory.is_dir() and os.access(paths.data_directory, os.R_OK),
            "The local data folder is missing or cannot be read.",
        ),
        (
            paths.outputs.is_dir() and os.access(paths.outputs, os.R_OK),
            "The saved research outputs folder is missing or cannot be read.",
        ),
        (
            paths.database.is_file() and os.access(paths.database, os.R_OK),
            "The local Department of Labor evidence database is missing or cannot be read.",
        ),
        (
            streamlit_checker(),
            "Streamlit is not installed in the project's .venv environment. "
            "Run the dependency installation steps in README.md.",
        ),
    ]
    for passed, message in checks:
        if not passed:
            raise LauncherError(message)


def build_streamlit_command(
    paths: LauncherPaths, *, app_port: int, token: str
) -> list[str]:
    """Return a shell-free argument list for the existing Streamlit dashboard."""

    return [
        str(paths.python),
        "-m",
        "streamlit",
        "run",
        str(paths.dashboard),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(app_port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--",
        "--db",
        str(paths.database),
        "--outputs",
        str(paths.outputs),
        "--launcher-token",
        token,
    ]


def port_is_available(port: int) -> bool:
    """Return whether a loopback TCP port can be bound right now."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def select_port_pair(
    *,
    ports: Sequence[int] = tuple(APP_PORTS),
    availability: Callable[[int], bool] = port_is_available,
) -> tuple[int, int]:
    """Select an app/control pair without disturbing occupied ports."""

    for app_port in ports:
        control_port = app_port + CONTROL_PORT_OFFSET
        if availability(app_port) and availability(control_port):
            return app_port, control_port
    raise LauncherError(
        "No local port is available between 8501 and 8510. Close another local "
        "development server or restart Windows, then try again."
    )


def _read_state(path: Path) -> LauncherState | None:
    """Read launcher state, treating corrupt or partial files as stale."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return LauncherState.from_mapping(value)
    except (FileNotFoundError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _write_state(path: Path, state: LauncherState) -> None:
    """Atomically replace state so another launch never reads partial JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _remove_state(path: Path) -> None:
    """Remove stale state without touching any process or application data."""

    try:
        path.unlink()
    except FileNotFoundError:
        pass


@contextmanager
def launcher_lock(path: Path) -> Iterator[None]:
    """Serialize short start/stop decisions with a one-byte Windows file lock."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        yield
    finally:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        handle.close()


def send_control_request(
    control_port: int,
    token: str,
    action: str,
    *,
    timeout: float = 1.0,
) -> dict[str, Any] | None:
    """Send one authenticated JSON request to a launcher-owned controller."""

    request = json.dumps({"token": token, "action": action}).encode("utf-8") + b"\n"
    try:
        with socket.create_connection(("127.0.0.1", control_port), timeout=timeout) as client:
            client.settimeout(timeout)
            client.sendall(request)
            response = bytearray()
            while len(response) < 16_384:
                block = client.recv(4096)
                if not block:
                    break
                response.extend(block)
                if b"\n" in block:
                    break
        line = bytes(response).splitlines()[0]
        value = json.loads(line.decode("utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, IndexError, json.JSONDecodeError):
        return None


def streamlit_is_healthy(app_port: int, *, timeout: float = 1.0) -> bool:
    """Check Streamlit's loopback health endpoint without using a proxy."""

    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(
            f"http://127.0.0.1:{app_port}/_stcore/health", timeout=timeout
        ) as response:
            return response.status == 200 and response.read(32).strip() == b"ok"
    except OSError:
        return False


def managed_instance_is_ready(
    state: LauncherState,
    *,
    control_request: Callable[[int, str, str], dict[str, Any] | None] = send_control_request,
    health_check: Callable[[int], bool] = streamlit_is_healthy,
) -> bool:
    """Require both authenticated controller identity and Streamlit health."""

    response = control_request(state.control_port, state.token, "ping")
    return bool(
        response
        and response.get("app_id") == APP_ID
        and response.get("project_root") == state.project_root
        and response.get("app_port") == state.app_port
        and response.get("status") == "running"
        and health_check(state.app_port)
    )


def _spawn_supervisor(
    paths: LauncherPaths,
    *,
    app_port: int,
    control_port: int,
    token: str,
) -> SpawnedProcess:
    """Start a hidden controller process using this exact launcher module."""

    command = [
        str(paths.pythonw),
        str(paths.launcher_script),
        "serve",
        "--project-root",
        str(paths.project_root),
        "--app-port",
        str(app_port),
        "--control-port",
        str(control_port),
        "--token",
        token,
    ]
    flags = 0
    if os.name == "nt":
        flags = (
            subprocess.CREATE_NO_WINDOW
            | subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    environment = os.environ.copy()
    # Desktop applications inherit an environment snapshot. If the user added
    # Brave configuration after opening the desktop parent, resolve it from the
    # Windows user scope and pass it only through the private child environment.
    resolved_brave_key = _brave_key_for_child()
    if resolved_brave_key and not environment.get("BRAVE_SEARCH_API_KEY"):
        environment["BRAVE_SEARCH_API_KEY"] = resolved_brave_key
    return subprocess.Popen(
        command,
        cwd=paths.project_root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        creationflags=flags,
        env=environment,
    )


def _wait_until_ready(
    state: LauncherState,
    process: SpawnedProcess,
    *,
    timeout_seconds: float,
    control_request: Callable[[int, str, str], dict[str, Any] | None] = send_control_request,
    health_check: Callable[[int], bool] = streamlit_is_healthy,
) -> bool:
    """Wait only for the bounded local startup window."""

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        response = control_request(state.control_port, state.token, "ping")
        if response and response.get("status") == "failed":
            return False
        if managed_instance_is_ready(
            state, control_request=control_request, health_check=health_check
        ):
            return True
        time.sleep(0.2)
    return False


def start_explorer(
    paths: LauncherPaths,
    *,
    open_browser: bool = True,
    opener: Callable[[str], Any] = lambda url: webbrowser.open(url, new=2),
    ports: Sequence[int] = tuple(APP_PORTS),
    availability: Callable[[int], bool] = port_is_available,
    instance_checker: Callable[[LauncherState], bool] = managed_instance_is_ready,
    spawner: Callable[..., SpawnedProcess] = _spawn_supervisor,
    ready_waiter: Callable[[LauncherState, SpawnedProcess], bool] | None = None,
    timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
) -> LaunchResult:
    """Reuse one verified instance or start exactly one new local server.

    Side effects include creating minimal state under ``.tmp``, starting the
    hidden controller/Streamlit processes, and optionally opening the default
    browser. No project database or output file is opened for writing.
    """

    validate_installation(paths)
    with launcher_lock(paths.lock_file):
        current = _read_state(paths.state_file)
        if current and instance_checker(current):
            url = f"http://127.0.0.1:{current.app_port}"
            if open_browser:
                opener(url)
            return LaunchResult("reused", url, current.app_port)
        if current:
            # A failed authenticated ping makes this record stale. Never kill
            # its PID or the process currently using its old port.
            _remove_state(paths.state_file)

        attempted_ports: set[int] = set()
        while len(attempted_ports) < len(ports):
            remaining = [port for port in ports if port not in attempted_ports]
            app_port, control_port = select_port_pair(
                ports=remaining, availability=availability
            )
            attempted_ports.add(app_port)
            token = secrets.token_urlsafe(24)
            process = spawner(
                paths,
                app_port=app_port,
                control_port=control_port,
                token=token,
            )
            state = LauncherState(
                app_id=APP_ID,
                project_root=str(paths.project_root),
                app_port=app_port,
                control_port=control_port,
                token=token,
                supervisor_pid=process.pid,
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            waiter = ready_waiter or (
                lambda candidate, child: _wait_until_ready(
                    candidate, child, timeout_seconds=timeout_seconds
                )
            )
            if waiter(state, process):
                _write_state(paths.state_file, state)
                url = f"http://127.0.0.1:{app_port}"
                if open_browser:
                    opener(url)
                return LaunchResult("started", url, app_port)
            if process.poll() is None:
                process.terminate()

        raise LauncherError(
            "Opportunity Explorer could not start on ports 8501–8510. No "
            "existing application or research data was changed."
        )


def stop_explorer(
    paths: LauncherPaths,
    *,
    control_request: Callable[[int, str, str], dict[str, Any] | None] = send_control_request,
) -> LaunchResult:
    """Stop only the instance authenticated by the launcher's saved token."""

    with launcher_lock(paths.lock_file):
        state = _read_state(paths.state_file)
        if state is None:
            _remove_state(paths.state_file)
            return LaunchResult("already_stopped", None, None)
        response = control_request(state.control_port, state.token, "stop")
        if not (
            response
            and response.get("app_id") == APP_ID
            and response.get("project_root") == state.project_root
            and response.get("app_port") == state.app_port
            and response.get("status") == "stopped"
        ):
            # The control identity is gone or different. The record is stale;
            # deleting it is safe, while killing its saved PID would not be.
            _remove_state(paths.state_file)
            return LaunchResult("stale_record", None, state.app_port)
        _remove_state(paths.state_file)
        return LaunchResult("stopped", None, state.app_port)


def _reply(connection: socket.socket, value: dict[str, Any]) -> None:
    """Write one bounded JSON-line control response."""

    connection.sendall(json.dumps(value).encode("utf-8") + b"\n")


def serve_controller(
    paths: LauncherPaths,
    *,
    app_port: int,
    control_port: int,
    token: str,
) -> int:
    """Own one Streamlit child until an authenticated stop request arrives."""

    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    command = build_streamlit_command(paths, app_port=app_port, token=token)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        server.bind(("127.0.0.1", control_port))
        server.listen(4)
        server.settimeout(0.5)
        child = subprocess.Popen(
            command,
            cwd=paths.project_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=flags,
        )
        failed_since: float | None = None
        try:
            while True:
                return_code = child.poll()
                if return_code is not None and failed_since is None:
                    failed_since = time.monotonic()
                if failed_since is not None and time.monotonic() - failed_since > 10:
                    return return_code or 1
                try:
                    connection, _ = server.accept()
                except socket.timeout:
                    continue
                with connection:
                    connection.settimeout(1.0)
                    try:
                        raw = connection.recv(16_384).splitlines()[0]
                        request = json.loads(raw.decode("utf-8"))
                    except (OSError, ValueError, IndexError, json.JSONDecodeError):
                        _reply(connection, {"app_id": APP_ID, "status": "invalid"})
                        continue
                    if request.get("token") != token:
                        _reply(connection, {"app_id": APP_ID, "status": "unauthorized"})
                        continue
                    status = "running" if child.poll() is None else "failed"
                    identity = {
                        "app_id": APP_ID,
                        "project_root": str(paths.project_root),
                        "app_port": app_port,
                        "status": status,
                    }
                    if request.get("action") == "ping":
                        _reply(connection, identity)
                    elif request.get("action") == "stop":
                        if child.poll() is None:
                            child.terminate()
                            try:
                                child.wait(timeout=8)
                            except subprocess.TimeoutExpired:
                                child.kill()
                                child.wait(timeout=3)
                        # Reply only after the owned child exits so a successful
                        # stop also means the application port is being released.
                        _reply(connection, {**identity, "status": "stopped"})
                        return 0
                    else:
                        _reply(connection, {**identity, "status": "invalid"})
        finally:
            if child.poll() is None:
                child.terminate()


def show_message(message: str, *, error: bool = False) -> None:
    """Display errors even when the launcher runs without a console window."""

    if os.name == "nt":
        import ctypes

        flags = 0x10 if error else 0x40
        ctypes.windll.user32.MessageBoxW(None, message, APP_TITLE, flags)
    else:
        print(message, file=sys.stderr if error else sys.stdout)


def _parser() -> argparse.ArgumentParser:
    """Build the small internal command-line interface used by VBScript."""

    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("action", choices=("start", "stop", "serve"))
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--app-port", type=int)
    parser.add_argument("--control-port", type=int)
    parser.add_argument("--token")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run start, stop, or the hidden controller and report friendly errors."""

    args = _parser().parse_args(argv)
    paths = launcher_paths(resolve_project_root(args.project_root))
    try:
        if args.action == "serve":
            if args.app_port is None or args.control_port is None or not args.token:
                raise LauncherError("The internal launcher request is incomplete.")
            return serve_controller(
                paths,
                app_port=args.app_port,
                control_port=args.control_port,
                token=args.token,
            )
        if args.action == "start":
            start_explorer(paths, open_browser=not args.no_browser)
            return 0
        result = stop_explorer(paths)
        if not args.quiet:
            if result.action == "stopped":
                show_message("Opportunity Explorer has been stopped.")
            elif result.action == "stale_record":
                show_message(
                    "Opportunity Explorer was not running. A stale launcher record was cleared."
                )
            else:
                show_message("Opportunity Explorer is already stopped.")
        return 0
    except LauncherError as exc:
        if args.quiet:
            print(str(exc), file=sys.stderr)
        else:
            show_message(str(exc), error=True)
        return 1
    except Exception:
        message = (
            "Opportunity Explorer could not start because of an unexpected local "
            "configuration problem. Review the Windows troubleshooting steps in README.md."
        )
        if args.quiet:
            print(message, file=sys.stderr)
        else:
            show_message(message, error=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
