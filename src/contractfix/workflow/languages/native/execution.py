"""Execute native commands in the task's already-prepared benchmark image."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import uuid

from contractfix.contracts.execution import ExecutionSettings, Executor, execute
from contractfix.contracts.core import digest


JAVA_BUILD_TOOLS = frozenset({"./gradlew", "./mvnw", "mvn", "mvnd", "ant"})


class NativeExecutor(Executor):
    """Reuse pinned image identity, omitting Python-only runtime overlays."""

    def __init__(self, image: str, *, timeout: float = 120.0,
                 allow_build_network: bool = False,
                 preserve_image_testbed: bool = False):
        settings = ExecutionSettings(kind="docker", image=image, timeout=timeout,
                                     pids_limit=1024 if preserve_image_testbed else 256,
                                     memory="8g" if preserve_image_testbed else "6g")
        super().__init__(settings)
        self.allow_build_network = allow_build_network
        self.preserve_image_testbed = preserve_image_testbed

    def identity(self) -> dict:
        identity = super().identity()
        if "native_build_network_policy" not in identity["runtime"]:
            body = {
                "settings": identity["settings"],
                "runtime": {
                    **identity["runtime"],
                    "native_build_network_policy": (
                        "java-toolchain-only/1" if self.allow_build_network else "none/1"
                    ),
                    **({"preserve_image_testbed": True,
                        "native_run_user": "host-uid-gid/1"}
                       if self.preserve_image_testbed else {}),
                },
            }
            self._identity = {**body, "sha256": digest(body)}
        return self._identity

    def run(self, command: list[str], work: Path, package_root: Path,
            bundle_path: Path, output_dir: Path, logfile: Path) -> tuple[int, str | None]:
        image_id = self.identity()["runtime"]["image_id"]
        name = "contractfix-native-" + uuid.uuid4().hex
        output_dir.mkdir(parents=True, exist_ok=True)
        workdir = "/opt/contractfix-work" if self.preserve_image_testbed else "/testbed"
        args = [
            "docker", "run", "--rm", "--pull=never", "--name", name,
            "--network=" + (
                "bridge" if self.allow_build_network and command[0] in JAVA_BUILD_TOOLS
                else "none"
            ),
            "--cap-drop=ALL", "--security-opt=no-new-privileges",
            "--pids-limit", str(self.settings.pids_limit),
            "--memory", self.settings.memory, "--cpus", str(self.settings.cpus),
            "--tmpfs", "/tmp:rw,exec,nosuid,size=1g", "--workdir", workdir,
            "--mount", f"type=bind,src={work.resolve()},dst={workdir}",
            "--mount", f"type=bind,src={output_dir.resolve()},dst=/opt/contractfix-events",
            *(["--user", f"{os.getuid()}:{os.getgid()}", "--env", "HOME=/tmp"]
              if self.preserve_image_testbed else []),
            image_id, *command,
        ]
        try:
            return execute(args, work, os.environ.copy(), self.settings.timeout, logfile)
        finally:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
