import json
import sys
from pathlib import Path
from typing import Any
import ast


_SANDBOX_ROOT = Path(__file__).resolve().parent
if str(_SANDBOX_ROOT) not in sys.path:
    sys.path.insert(0, str(_SANDBOX_ROOT))


from meta_agent.runtime import validate_source
from meta_agent.runtime.source_validation import SAFE_BUILTINS

COMPONENTS: dict[str, Any] = {}

MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESULT_BYTES = 1024 * 1024
MAX_SOURCE_BYTES = 512 * 1024
MAX_COMPONENTS = 128


def send_response(request_id: str | None, status: str, **payload: Any) -> None:

    response = {
        "request_id": request_id,
        "status": status,
        **payload,
    }

    print(
        json.dumps(response, ensure_ascii=False, allow_nan=False),
        flush=True,
    )


def register_component(
    request: dict[str, Any],
) -> dict[str, Any]:

    component_id = request.get("component_id")
    source_code = request.get("source_code")
    entrypoint = request.get("entrypoint", "transform")

    if not isinstance(component_id, str):
        raise ValueError("component_id must be a string")
    
    if not component_id:
        raise ValueError("component_id is required")

    if component_id.startswith("_"):
        raise ValueError("component_id cannot start with an underscore")

    if (
        component_id in COMPONENTS
        and len(COMPONENTS) >= MAX_COMPONENTS
    ):
        raise ValueError(f"maximum number of registered components exceeded: {MAX_COMPONENTS}")

    if component_id in COMPONENTS:
        raise ValueError(f"component is already registered: {component_id}")
    
    if not isinstance(source_code, str):
        raise ValueError("source_code must be a string")

    source_size = len(source_code.encode("utf-8"))

    if source_size > MAX_SOURCE_BYTES:
        raise ValueError(f"source_code exceeds maximum size: {source_size} > {MAX_SOURCE_BYTES} bytes")
    
    if not isinstance(entrypoint, str):
        raise ValueError("entrypoint must be a string")

    validate_source(source_code, entrypoint)
    tree = ast.parse(source_code)

    namespace = {
        "__builtins__": SAFE_BUILTINS,
    }

    compiled = compile(
        tree,
        filename="<generated-component>",
        mode="exec"
    )

    exec(compiled, namespace, namespace)

    function = namespace.get(entrypoint)

    if not callable(function):
        raise ValueError("entrypoint must be a callable function")

    COMPONENTS[component_id] = function

    return {
        "component_id": component_id,
        "entrypoint": entrypoint,
    }


def invoke_component(
    request: dict[str, Any],
) -> dict[str, Any]:

    component_id = request.get("component_id")
    arguments = request.get("arguments", {})

    if component_id not in COMPONENTS:
        raise ValueError(f"unknown component {component_id}")

    if not isinstance(arguments, dict):
        raise ValueError("arguments must be a dictionary")

    function = COMPONENTS[component_id]

    result = function(**arguments)

    encoded_result = json.dumps(
        result,
        ensure_ascii=False,
        allow_nan=False,
    )

    result_size = len(encoded_result.encode("utf-8"))

    if result_size > MAX_RESULT_BYTES:
        raise ValueError(f"component result exceeds maximum size: {result_size} > {MAX_RESULT_BYTES} bytes")

    return {
        "result": result,
    }


def process_request(
    request: dict[str, Any],
) -> tuple[dict[str, Any], bool]:

    operation = request.get("operation")

    if operation == "ping":
        return {"worker": "ready"}, False

    if operation == "register":
        return register_component(request), False

    if operation == "invoke":
        return invoke_component(request), False

    if operation == "shutdown":
        return {}, True

    raise ValueError(
        f"unknown operation: {operation}"
    )


def main() -> None:
    while True:
        raw_line = sys.stdin.buffer.readline(
            MAX_REQUEST_BYTES + 1
        )

        if not raw_line:
            break

        if len(raw_line) > MAX_REQUEST_BYTES:
            send_response(
                None,
                "error",
                error_type="ValueError",
                message=(
                    "request exceeds maximum size: "
                    f"{MAX_REQUEST_BYTES} bytes"
                ),
            )
            break

        request_id = None

        try:
            line = raw_line.decode("utf-8")
            request = json.loads(line)

            if not isinstance(request, dict):
                raise ValueError(
                    "request must be a JSON object"
                )

            request_id = request.get("request_id")
            payload, should_stop = process_request(request)

            send_response(
                request_id,
                "success",
                **payload,
            )

            if should_stop:
                break

        except Exception as error:
            send_response(
                request_id,
                "error",
                error_type=type(error).__name__,
                message=str(error),
            )


if __name__ == "__main__":
    main()
