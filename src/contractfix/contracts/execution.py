"""Execution backends, deliberately separate from model orchestration and EC logic."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from .core import digest


@dataclass(frozen=True)
class ExecutionSettings:
    kind: str = "docker"
    image: str | None = None
    python: str = "/opt/miniconda3/envs/testbed/bin/python"
    repo_path: str = "/testbed"
    timeout: float = 120.0
    memory: str = "6g"
    cpus: float = 2.0
    pids_limit: int = 256
    environment: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {"local", "docker"}:
            raise ValueError("executor kind must be local or docker")
        if self.kind == "docker" and not self.image:
            raise ValueError("docker requires an existing historical task image")
        if self.timeout <= 0 or self.cpus <= 0 or self.pids_limit < 16:
            raise ValueError("positive resource budgets required")
        if not self.repo_path.startswith("/") or self.repo_path == "/":
            raise ValueError("container repository path must be absolute and non-root")
        if set(self.environment) - {"DJANGO_SETTINGS_MODULE", "MPLBACKEND", "TZ", "LANG", "LC_ALL"}:
            raise ValueError("only explicitly permitted non-secret environment variables may be passed")


def execute(command: list[str], cwd: Path, env: dict[str, str], timeout: float,
            logfile: Path) -> tuple[int, str | None]:
    """Bound wall time and stdout disk use; terminate the complete local process group."""
    with logfile.open("wb") as output:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + timeout
        error = None
        while process.poll() is None:
            if time.monotonic() >= deadline:
                error = "timeout"
            elif logfile.stat().st_size > 16 * 1024 * 1024:
                error = "command_log_budget_exceeded"
            if error:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                break
            time.sleep(0.025)
        return process.returncode, error


class Executor:
    """Pin a backend identity before either version runs; never silently pull an image."""

    def __init__(self, settings: ExecutionSettings):
        self.settings = settings
        self._identity: dict | None = None
        self._native_artifacts = None

    def identity(self) -> dict:
        if self._identity is None:
            if self.settings.kind == "docker":
                result = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}",
                                         self.settings.image], capture_output=True, text=True, timeout=30)
                if result.returncode:
                    raise RuntimeError("Docker/image unavailable; build or obtain the task environment first: "
                                       + result.stderr[-600:])
                runtime = {"image_id": result.stdout.strip(),
                           "native_artifact_policy": "pinned-image-binaries/1"}
            else:
                result = subprocess.run([self.settings.python, "-c",
                    "import json,sys; print(json.dumps({'python':sys.version,'executable':sys.executable}))"],
                    capture_output=True, text=True, timeout=30)
                if result.returncode:
                    raise RuntimeError("target Python could not start")
                runtime = json.loads(result.stdout)
            body = {"settings": asdict(self.settings), "runtime": runtime}
            self._identity = {**body, "sha256": digest(body)}
        return self._identity

    def _command(self, command: list[str]) -> list[str]:
        return [self.settings.python if item == "@python" else item for item in command]

    def run(self, command: list[str], work: Path, package_root: Path,
            bundle_path: Path, output_dir: Path, logfile: Path) -> tuple[int, str | None]:
        self.identity()
        output_dir.mkdir(parents=True, exist_ok=True)
        events_dir = output_dir / "events"
        events_dir.mkdir(parents=True, exist_ok=True)
        source_layouts = [name for name in ("src", "lib") if (work / name).is_dir()
                          and not (work / name / "__init__.py").exists()]
        if self.settings.kind == "local":
            # CLI requires explicit --allow-local-execution. This is NOT a sandbox.
            env = {name: os.environ[name] for name in ("PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH")
                   if name in os.environ}
            home = output_dir / "home"
            home.mkdir(exist_ok=True)
            env.update(self.settings.environment)
            pythonpath = os.pathsep.join([str(package_root), str(work),
                                         *(str(work / name) for name in source_layouts)])
            env.update(HOME=str(home), PYTHONPATH=pythonpath,
                PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="42",
                CONTRACTFIX_EC_BUNDLE=str(bundle_path), CONTRACTFIX_EC_EVENTS=str(events_dir),
                CONTRACTFIX_EC_REPO=str(work))
            return execute(self._command(command), work, env, self.settings.timeout, logfile)
        # SWE-bench images commonly run as a non-host UID. The directory is a
        # fresh, per-execution temporary output mount containing no inputs.
        events_dir.chmod(0o777)
        name = "contractfix-" + uuid.uuid4().hex
        args = ["docker", "run", "--rm", "--pull=never", "--name", name,
                "--network=none", "--read-only", "--cap-drop=ALL",
                "--security-opt=no-new-privileges", "--pids-limit", str(self.settings.pids_limit),
                "--memory", self.settings.memory, "--cpus", str(self.settings.cpus),
                "--tmpfs", "/tmp:rw,exec,nosuid,size=1g", "--workdir", self.settings.repo_path]
        for source, target, read_only in (
            (work, self.settings.repo_path, False),
            (package_root, "/opt/contractfix-runtime", True),
            (bundle_path, "/opt/contractfix-bundle.json", True),
            (output_dir, "/opt/contractfix-events", False),
        ):
            args += ["--mount", f"type=bind,src={source},dst={target}" + (",readonly" if read_only else "")]
        env = {**self.settings.environment, "HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "42", "PYTHONPATH": ":".join(
                ["/opt/contractfix-runtime", self.settings.repo_path,
                 *(self.settings.repo_path + "/" + name for name in source_layouts)]),
            "CONTRACTFIX_EC_BUNDLE": "/opt/contractfix-bundle.json",
            "CONTRACTFIX_EC_EVENTS": "/opt/contractfix-events/events",
            "CONTRACTFIX_EC_REPO": self.settings.repo_path}
        for key, value in env.items():
            args += ["--env", key + "=" + value]
        args += [self.identity()["runtime"]["image_id"], *self._command(command)]
        try:
            from .native_artifacts import NativeArtifacts

            if self._native_artifacts is None:
                self._native_artifacts = NativeArtifacts(
                    self.identity()["runtime"]["image_id"], self.settings.python,
                    self.settings.repo_path,
                )
            with self._native_artifacts.overlay(work):
                return execute(args, work, os.environ.copy(), self.settings.timeout, logfile)
        finally:
            # Killing docker's client alone does not guarantee container termination.
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
