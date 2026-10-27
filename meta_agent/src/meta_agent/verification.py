"""Проверка generated-компонентов в Docker sandbox и учёт verified.json."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from meta_agent.config.schema import AgentConfig
from meta_agent.generated_paths import (
    GeneratedPaths,
    resolve_generated_paths,
)
from meta_agent.generated_tools import GeneratedToolSpec
from meta_agent.generated_tools import load_generated_tool
from meta_agent.runtime.docker import DockerRuntime
from meta_agent.runtime.source_validation import validate_source
from meta_agent.sandbox import (
    DockerSandboxExecutor,
    SandboxError,
    SandboxExecutionError,
    SandboxProtocolError,
    SandboxStartError,
    SandboxTimeoutError,
)
from meta_agent.smoke_inputs import (
    SMOKE_SEED,
    assert_tool_schema,
    invalid_tool_args,
    invalid_transform_data,
    valid_context_transform_data,
    valid_observation_transform_data,
    valid_tool_args,
)
from meta_agent.transforms import TransformSpec
from meta_agent.transforms import load_transform


DEFAULT_VERIFIED_HASHES_PATH = (
    resolve_generated_paths().verified_path
)

ValidVerdict = Literal["ok", "exception", "timeout_or_crash"]
InvalidVerdict = Literal[
    "expected_exception",
    "timeout_or_crash",
    "unexpected_ok",
]


@dataclass(frozen=True)
class ComponentVerifyResult:
    """Результат verify одного компонента."""

    component_id: str
    valid: ValidVerdict
    invalid: InvalidVerdict
    accepted: bool


def _load_verified_hashes(verified_path: Path) -> dict[str, str]:
    if not verified_path.is_file():
        return {}

    with verified_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("verified.json must be a JSON object")

    return {
        str(key): str(value)
        for key, value in data.items()
    }


def _save_verified_hashes(
    verified: dict[str, str],
    verified_path: Path,
) -> None:
    verified_path.parent.mkdir(exist_ok=True, parents=True)

    content = (
        json.dumps(
            dict(sorted(verified.items())),
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    tmp_path = verified_path.with_name(
        verified_path.name + ".tmp"
    )
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, verified_path)


def _record_verified_hash(
    component_id: str,
    source_hash: str,
    *,
    verified_path: Path,
) -> None:
    verified = _load_verified_hashes(verified_path)
    verified[component_id] = source_hash
    _save_verified_hashes(verified, verified_path)


def _check_source_hash(source_code: str, expected_hash: str) -> None:
    computed_hash = sha256(source_code.encode("utf-8")).hexdigest()
    if computed_hash != expected_hash:
        raise ValueError("Source code hash mismatch")


def check_verified_hash(
    component_id: str,
    source_hash: str,
    *,
    verified_path: str | Path | None = None,
) -> None:
    """Проверяет, что компонент зарегистрирован в verified.json.

    Args:
        component_id: Идентификатор вида ``tool:...`` / ``transform:...``.
        source_hash: Ожидаемый SHA-256 исходника.
        verified_path: Путь к ``verified.json``; по умолчанию
            ``meta_agent/generated/verified.json``.
    """
    path = (
        Path(verified_path).resolve()
        if verified_path is not None
        else DEFAULT_VERIFIED_HASHES_PATH
    )
    verified = _load_verified_hashes(path)
    expected_hash = verified.get(component_id)

    if expected_hash != source_hash:
        raise ValueError(
            f"component is not verified: {component_id}"
        )


def _is_crash_or_timeout(error: BaseException) -> bool:
    return isinstance(
        error,
        (
            SandboxTimeoutError,
            SandboxProtocolError,
            SandboxStartError,
            TimeoutError,
        ),
    )


def _classify_valid_call(invoke: Any) -> tuple[ValidVerdict, Any]:
    """Вызывает invoke; возвращает (вердикт, результат|None)."""
    try:
        result = invoke()
        return "ok", result
    except SandboxExecutionError:
        return "exception", None
    except Exception as error:
        if _is_crash_or_timeout(error):
            return "timeout_or_crash", None
        if isinstance(error, SandboxError):
            return "timeout_or_crash", None
        return "exception", None


def _classify_invalid_call(invoke: Any) -> InvalidVerdict:
    try:
        invoke()
        return "unexpected_ok"
    except SandboxExecutionError:
        return "expected_exception"
    except Exception as error:
        if _is_crash_or_timeout(error):
            return "timeout_or_crash"
        if isinstance(error, SandboxError):
            return "timeout_or_crash"
        return "expected_exception"


def _accept_verdicts(
    valid: ValidVerdict,
    invalid: InvalidVerdict,
) -> bool:
    return valid == "ok" and invalid == "expected_exception"


def _check_tool_output(
    result: Any,
    output_schema: dict[str, Any],
) -> None:
    """Лёгкая проверка результата: JSON-able; для dict — ключи схемы."""
    json.dumps(result)
    if output_schema and isinstance(result, dict):
        missing = [
            key for key in output_schema if key not in result
        ]
        if missing:
            raise TypeError(
                "tool result missing output_schema keys: "
                + ", ".join(sorted(missing))
            )


def verify_transform(
    spec: TransformSpec,
    *,
    valid_data: dict[str, Any],
    verified_path: str | Path | None = None,
) -> ComponentVerifyResult:
    """Верифицирует transform: 1 valid + 1 invalid smoke в sandbox.

    Args:
        spec: Спецификация transform.
        valid_data: Валидный ``data`` из observation/context contract.
        verified_path: Куда писать ``verified.json`` при accept.

    Returns:
        Вердикт компонента.

    Raises:
        ValueError: Некорректный spec до sandbox.
        RuntimeError: Reject по вердиктам (hash не пишется).
    """
    if not spec.transform_id:
        raise ValueError("transform_id must not be empty")
    if not spec.source_code:
        raise ValueError("source_code must not be empty")
    if not spec.entrypoint:
        raise ValueError("entrypoint must not be empty")

    _check_source_hash(spec.source_code, spec.source_hash)
    validate_source(spec.source_code, spec.entrypoint)

    component_id = f"transform:{spec.transform_id}"
    path = (
        Path(verified_path).resolve()
        if verified_path is not None
        else DEFAULT_VERIFIED_HASHES_PATH
    )

    with DockerSandboxExecutor() as executor:
        with DockerRuntime(executor) as runtime:
            runtime.register(
                component_id=component_id,
                source_code=spec.source_code,
                entrypoint=spec.entrypoint,
            )

            valid_verdict, result = _classify_valid_call(
                lambda: runtime.invoke(
                    component_id=component_id,
                    arguments={"data": valid_data},
                )
            )
            if valid_verdict == "ok":
                try:
                    if not isinstance(result, dict):
                        raise TypeError(
                            "transform result must be a dictionary"
                        )
                    json.dumps(result)
                except (TypeError, ValueError):
                    valid_verdict = "exception"

            invalid_verdict = _classify_invalid_call(
                lambda: runtime.invoke(
                    component_id=component_id,
                    arguments={"data": invalid_transform_data()},
                )
            )

    accepted = _accept_verdicts(valid_verdict, invalid_verdict)
    outcome = ComponentVerifyResult(
        component_id=component_id,
        valid=valid_verdict,
        invalid=invalid_verdict,
        accepted=accepted,
    )

    if not accepted:
        raise RuntimeError(
            f"verify rejected {component_id}: "
            f"valid={valid_verdict}, invalid={invalid_verdict}"
        )

    _record_verified_hash(
        component_id,
        spec.source_hash,
        verified_path=path,
    )
    return outcome


def verify_generated_tool(
    spec: GeneratedToolSpec,
    *,
    verified_path: str | Path | None = None,
) -> ComponentVerifyResult:
    """Верифицирует tool: schema gate + 1 valid + 1 invalid smoke.

    Args:
        spec: Спецификация tool.
        verified_path: Куда писать ``verified.json`` при accept.

    Returns:
        Вердикт компонента.

    Raises:
        ValueError: Некорректный spec / схема.
        RuntimeError: Reject по вердиктам.
    """
    if not spec.tool_id:
        raise ValueError("tool_id must not be empty")
    if not spec.name:
        raise ValueError("name must not be empty")
    if not spec.description:
        raise ValueError("description must not be empty")
    if not spec.entrypoint:
        raise ValueError("entrypoint must not be empty")
    if not spec.output_schema:
        raise ValueError("output_schema must not be empty")
    if not spec.source_code:
        raise ValueError("source_code must not be empty")

    _check_source_hash(spec.source_code, spec.source_hash)
    validate_source(spec.source_code, spec.entrypoint)
    assert_tool_schema(spec)

    component_id = f"tool:{spec.tool_id}"
    path = (
        Path(verified_path).resolve()
        if verified_path is not None
        else DEFAULT_VERIFIED_HASHES_PATH
    )

    valid_args = valid_tool_args(spec.input_schema)
    invalid_args = invalid_tool_args(spec.input_schema)

    with DockerSandboxExecutor() as executor:
        with DockerRuntime(executor) as runtime:
            runtime.register(
                component_id=component_id,
                source_code=spec.source_code,
                entrypoint=spec.entrypoint,
                timeout_sec=spec.timeout_sec,
            )

            valid_verdict, result = _classify_valid_call(
                lambda: runtime.invoke(
                    component_id=component_id,
                    arguments=valid_args,
                    timeout_sec=spec.timeout_sec,
                )
            )
            if valid_verdict == "ok":
                try:
                    _check_tool_output(result, spec.output_schema)
                except (TypeError, ValueError):
                    valid_verdict = "exception"

            invalid_verdict = _classify_invalid_call(
                lambda: runtime.invoke(
                    component_id=component_id,
                    arguments=invalid_args,
                    timeout_sec=spec.timeout_sec,
                )
            )

    accepted = _accept_verdicts(valid_verdict, invalid_verdict)
    outcome = ComponentVerifyResult(
        component_id=component_id,
        valid=valid_verdict,
        invalid=invalid_verdict,
        accepted=accepted,
    )

    if not accepted:
        raise RuntimeError(
            f"verify rejected {component_id}: "
            f"valid={valid_verdict}, invalid={invalid_verdict}"
        )

    _record_verified_hash(
        component_id,
        spec.source_hash,
        verified_path=path,
    )
    return outcome


def verify_config_components(
    config: AgentConfig,
    paths: GeneratedPaths,
) -> tuple[list[str], list[ComponentVerifyResult]]:
    """Верифицирует только компоненты, на которые ссылается config.

    Args:
        config: ``AgentConfig``.
        paths: Разрешённые пути generated-root.

    Returns:
        Пара (список accepted component_id, детальные results).

    Raises:
        RuntimeError: Любой компонент не принят.
    """
    results: list[ComponentVerifyResult] = []
    verified_ids: list[str] = []

    transform_roles: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()

    context_id = config.context_contract.transform
    if context_id is not None and context_id not in seen:
        seen.add(context_id)
        transform_roles.append(
            (
                context_id,
                valid_context_transform_data(
                    config.context_contract.sections
                ),
            )
        )

    observation_id = config.observation_contract.transform
    if observation_id is not None and observation_id not in seen:
        seen.add(observation_id)
        transform_roles.append(
            (
                observation_id,
                valid_observation_transform_data(
                    config.observation_contract.included_fields
                ),
            )
        )
    elif (
        observation_id is not None
        and observation_id == context_id
    ):
        # Тот же transform на оба контракта: уже проверен на context.
        pass

    for transform_id, valid_data in transform_roles:
        if (
            not transform_id
            or Path(transform_id).name != transform_id
            or transform_id in {".", ".."}
        ):
            raise ValueError(
                f"invalid transform ID: {transform_id}"
            )

        transform_path = (
            paths.transforms_dir / f"{transform_id}.json"
        )
        if not transform_path.is_file():
            raise FileNotFoundError(
                f"transform file does not exist: {transform_path}"
            )

        spec = load_transform(transform_path)
        if spec.transform_id != transform_id:
            raise ValueError(
                "transform ID does not match filename: "
                f"expected {transform_id}, "
                f"found {spec.transform_id}"
            )

        outcome = verify_transform(
            spec,
            valid_data=valid_data,
            verified_path=paths.verified_path,
        )
        results.append(outcome)
        verified_ids.append(outcome.component_id)

    tool_ids = list(config.generated_tool_ids)
    if len(tool_ids) != len(set(tool_ids)):
        raise ValueError("duplicate generated tool IDs")

    for tool_id in tool_ids:
        if (
            not tool_id
            or Path(tool_id).name != tool_id
            or tool_id in {".", ".."}
        ):
            raise ValueError(
                f"invalid generated tool ID: {tool_id}"
            )

        tool_path = paths.tools_dir / f"{tool_id}.json"
        if not tool_path.is_file():
            raise FileNotFoundError(
                "generated tool file does not exist: "
                f"{tool_path}"
            )

        spec = load_generated_tool(tool_path)
        if spec.tool_id != tool_id:
            raise ValueError(
                "generated tool ID does not match filename: "
                f"expected {tool_id}, found {spec.tool_id}"
            )

        outcome = verify_generated_tool(
            spec,
            verified_path=paths.verified_path,
        )
        results.append(outcome)
        verified_ids.append(outcome.component_id)

    return verified_ids, results
