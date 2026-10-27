import json
import os
import selectors
import subprocess
import threading
import uuid
from typing import Any


MAX_REQUEST_BYTES = 1024 * 1024


class SandboxError(RuntimeError):
    pass


class SandboxStartError(SandboxError):
    pass


class SandboxProtocolError(SandboxError):
    pass


class SandboxExecutionError(SandboxError):
    pass


class SandboxTimeoutError(SandboxError):
    pass


class DockerSandboxExecutor:
    
    def __init__(
        self,
        image: str = "meta-agent-sandbox:latest",
        timeout_sec: float = 2.0,
        memory: str = "128m",
        cpus: float = 0.5,
        pids_limit: int = 32
    ):
        self.image = image
        self.timeout_sec = timeout_sec
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit

        self.container_name = (
            f"meta-agent-sandbox-{uuid.uuid4().hex[:12]}"
        )

        self.process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self.process is not None:
            raise SandboxStartError("sandbox already running")

        command = [
            "docker",
            "run",
            "--rm",
            "--interactive",
            "--name",
            self.container_name,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--memory",
            self.memory,
            "--memory-swap",
            self.memory,
            "--cpus",
            str(self.cpus),
            "--pids-limit",
            str(self.pids_limit),
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=16m",
            "--user",
            "10001:10001",
            self.image,
        ]

        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=os.environ.copy(),
            )

            response = self.request(
                operation="ping",
                timeout_sec=10.0
            )

            if response["worker"] != "ready":
                raise SandboxStartError("sandbox worker did not become ready")

        except Exception as error:
            self._terminate()
            raise

    def request(
        self,
        operation: str,
        timeout_sec: float | None = None,
        **payload: Any,
    ) -> dict[str, Any]:

        process = self.process

        if process is None:
            raise SandboxStartError("sandbox not running")

        if process.poll() is not None:
            raise SandboxProtocolError("sandbox process has terminated")

        request_id = uuid.uuid4().hex

        request = {
            "request_id": request_id,
            "operation": operation,
            **payload,
        }

        encoded = json.dumps(
            request,
            ensure_ascii=False,
            allow_nan=False
        )

        encoded_size = len(encoded.encode("utf-8"))

        if encoded_size > MAX_REQUEST_BYTES:
            raise SandboxProtocolError(f"sandbox request exceeds maximum size {encoded_size} > {MAX_REQUEST_BYTES} bytes")

        timeout = (
            self.timeout_sec if timeout_sec is None else timeout_sec
        )

        with self._lock:
            assert process.stdin is not None
            assert process.stdout is not None

            try:
                process.stdin.write(encoded + "\n")
                process.stdin.flush()
            
            except (BrokenPipeError, OSError) as error:
                raise SandboxProtocolError("sandbox input pipe is closed") from error

            selector = selectors.DefaultSelector()
            selector.register(
                process.stdout,
                selectors.EVENT_READ,
            )

            try:
                events = selector.select(timeout)

                if not events:
                    self._terminate()

                    raise SandboxTimeoutError(f"sandbox operation timed out after {timeout} seconds")

                line = process.stdout.readline()
            
            finally:
                selector.close()
        
        if not line:
            stderr = self._read_stderr()
            raise SandboxProtocolError(f"sandbox terminated without a response: {stderr}")
        
        try:
            response = json.loads(line)
        
        except json.JSONDecodeError as error:
            raise SandboxProtocolError(f"sandbox returned invalid JSON {line!r}") from error

        if response.get("request_id") != request_id:
            raise SandboxProtocolError("sandbox response request ID mismatch")

        if response.get("status") != "success":
            error_type = response.get("error_type", "SandboxExecutionError")
            message = response.get("message", "sandbox operation failed")

            raise SandboxExecutionError(f"{error_type}: {message}")

        return response


    def register(
        self,
        component_id: str,
        source_code: str,
        entrypoint: str = "transform",
        timeout_sec: float | None = None,
    ) -> None:
        del timeout_sec
        self.request(
            operation="register",
            component_id=component_id,
            source_code=source_code,
            entrypoint=entrypoint,
        )


    def invoke(
        self,
        component_id: str,
        arguments: dict[str, Any],
        timeout_sec: float | None = None,
    ) -> Any:
        response = self.request(
            operation="invoke",
            component_id=component_id,
            arguments=arguments,
            timeout_sec=timeout_sec,
        )

        return response["result"]

    def stop(self) -> None:
        process = self.process

        if process is None:
            return

        try:
            if process.poll() is None:
                self.request(
                    operation="shutdown",
                    timeout_sec=1.0
                )

            process.wait(timeout=2.0)
        
        except Exception:
            self._terminate()

        finally:
            self.process = None

    def _terminate(self) -> None:
        process = self.process

        if process is None:
            return

        try:
            subprocess.run(
                [
                    "docker",
                    "kill",
                    self.container_name,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5.0,
                check=False,
                env=os.environ.copy(),
            )

        except (
            OSError,
            subprocess.TimeoutExpired,
        ):
            pass

        try:
            process.wait(timeout=2.0)

        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)

        self.process = None

    def _read_stderr(self) -> str:
        process = self.process

        if (
            process is None
            or process.stderr is None
        ):
            return ""

        if process.poll() is None:
            return ""

        return process.stderr.read().strip()[:2000]

    def __enter__(self) -> "DockerSandboxExecutor":
        self.start()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback
    ) -> None:

        self.stop()
