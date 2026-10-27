import hashlib
from pathlib import Path
from dataclasses import asdict

import os
from collections.abc import Callable
from typing import Any

import httpx

from agents.react.agent import ReActAgent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_openrouter import ChatOpenRouter
from openrouter import OpenRouter
from openrouter.utils.retries import BackoffStrategy, RetryConfig

from meta_agent.config.policy import ConfigurableAgentPolicy
from meta_agent.config.schema import AgentConfig
from meta_agent.generated_paths import (
    GeneratedPaths,
    resolve_generated_paths,
)
from meta_agent.generated_tools import (
    GeneratedToolRegistry,
    build_langchain_tool,
    load_generated_tool,
)
from meta_agent.runtime import CodeRuntime
from meta_agent.runtime.local import LocalRuntime
from meta_agent.transforms import TransformRegistry, load_transform


ModelFactory = Callable[[AgentConfig], BaseChatModel]


DEFAULT_TRANSFORM_DIR = (
    resolve_generated_paths().transforms_dir
)

DEFAULT_GENERATED_TOOL_DIR = (
    resolve_generated_paths().tools_dir
)


def build_config_metadata(
    config: AgentConfig,
    *,
    transform_registry: TransformRegistry | None = None,
    generated_tool_registry: GeneratedToolRegistry | None = None,
) -> dict[str, Any]:

    prompt_hash = hashlib.sha256(
        config.system_prompt.encode("utf-8")
    ).hexdigest()

    transform_snapshots = (
        transform_registry.snapshot()
        if transform_registry is not None
        else []
    )

    generated_tool_snapshots = (
        generated_tool_registry.snapshot()
        if generated_tool_registry is not None
        else []
    )

    return {
        "agent_config_id": config.config_id,
        "prompt_hash": prompt_hash,
        "config_snapshot": asdict(config),
        "transform_snapshots": transform_snapshots,
        "generated_tool_snapshots": (
            generated_tool_snapshots
        ),
    }


def build_tool_options(
    config: AgentConfig,
) -> dict[str, dict[str, Any]]:

    tool_options: dict[str, dict[str, Any]] = {}

    for tool_spec in config.tool_specs:
        if tool_spec.name in tool_options:
            raise ValueError(f"duplicate tool name: {tool_spec.name}")
        
        tool_options[tool_spec.name] = {
            "enabled": tool_spec.enabled,
            "max_wait_ticks": tool_spec.max_wait_ticks,
            "kwargs_defaults": dict(tool_spec.kwargs_defaults),
            "description": tool_spec.description,
        }

    return tool_options


def build_transform_registry(
    config: AgentConfig,
    *,
    runtime: CodeRuntime,
    transform_directory: str | Path | None = None,
    verified_path: str | Path | None = None,
) -> TransformRegistry | None:
    """Собирает registry transforms, указанных в config.

    Args:
        config: Конфигурация агента.
        runtime: Runtime для исполнения кода.
        transform_directory: Каталог ``*.json`` transforms.
        verified_path: Путь к ``verified.json``.
    """
    transform_ids = [
        transform_id
        for transform_id in (
            config.context_contract.transform,
            config.observation_contract.transform,
        )
        if transform_id is not None
    ]

    # Сохраняем порядок и исключаем повторную
    # регистрацию одного transform.
    transform_ids = list(
        dict.fromkeys(transform_ids)
    )

    if not transform_ids:
        return None

    directory = (
        Path(transform_directory)
        if transform_directory is not None
        else DEFAULT_TRANSFORM_DIR
    )

    specs = []

    for transform_id in transform_ids:
        if (
            not transform_id
            or Path(transform_id).name
            != transform_id
            or transform_id in {".", ".."}
        ):
            raise ValueError(
                f"invalid transform ID: {transform_id}"
            )

        transform_path = (
            directory / f"{transform_id}.json"
        )

        if not transform_path.is_file():
            raise FileNotFoundError(
                "transform file does not exist: "
                f"{transform_path}"
            )

        spec = load_transform(transform_path)

        if spec.transform_id != transform_id:
            raise ValueError(
                "transform ID does not match filename: "
                f"expected {transform_id}, "
                f"found {spec.transform_id}"
            )

        specs.append(spec)

    from meta_agent.verification import check_verified_hash

    for spec in specs:
        check_verified_hash(
            f"transform:{spec.transform_id}",
            spec.source_hash,
            verified_path=verified_path,
        )

    registry = TransformRegistry(runtime)

    for spec in specs:
        registry.register(spec)

    return registry


def build_generated_tool_registry(
    config: AgentConfig,
    *,
    runtime: CodeRuntime,
    generated_tool_directory: str | Path | None = None,
    verified_path: str | Path | None = None,
) -> GeneratedToolRegistry | None:
    """Собирает registry generated tools из config.

    Args:
        config: Конфигурация агента.
        runtime: Runtime для исполнения кода.
        generated_tool_directory: Каталог ``*.json`` tools.
        verified_path: Путь к ``verified.json``.
    """
    tool_ids = list(
        config.generated_tool_ids
    )

    if not tool_ids:
        return None

    if len(tool_ids) != len(set(tool_ids)):
        raise ValueError(
            "duplicate generated tool IDs"
        )

    directory = (
        Path(generated_tool_directory)
        if generated_tool_directory is not None
        else DEFAULT_GENERATED_TOOL_DIR
    )

    specs = []

    # Сначала проверяем все файлы.
    # На этом этапе ничего не регистрируется.
    for tool_id in tool_ids:
        if (
            not tool_id
            or Path(tool_id).name != tool_id
            or tool_id in {".", ".."}
        ):
            raise ValueError(
                f"invalid generated tool ID: "
                f"{tool_id}"
            )

        tool_path = (
            directory / f"{tool_id}.json"
        )

        if not tool_path.is_file():
            raise FileNotFoundError(
                "generated tool file does not "
                f"exist: {tool_path}"
            )

        spec = load_generated_tool(
            tool_path
        )

        if spec.tool_id != tool_id:
            raise ValueError(
                "generated tool ID does not "
                "match filename: "
                f"expected {tool_id}, "
                f"found {spec.tool_id}"
            )

        specs.append(spec)

    from meta_agent.verification import check_verified_hash

    for spec in specs:
        check_verified_hash(
            f"tool:{spec.tool_id}",
            spec.source_hash,
            verified_path=verified_path,
        )

    registry = GeneratedToolRegistry(runtime)

    for spec in specs:
        registry.register(spec)

    return registry


def build_generated_tools(
    config: AgentConfig,
    *,
    registry: GeneratedToolRegistry | None,
) -> list[BaseTool]:

    if not config.generated_tool_ids:
        return []

    if registry is None:
        raise ValueError(
            "generated tools are configured, "
            "but generated tool registry "
            "is not provided"
        )

    tools: list[BaseTool] = []

    for tool_id in config.generated_tool_ids:
        spec = registry.get(tool_id)

        tool = build_langchain_tool(
            spec=spec,
            registry=registry,
        )

        tools.append(tool)

    return tools


def validate_tool_name_conflicts(
    config: AgentConfig,
    generated_tools: list[BaseTool],
) -> None:

    builtin_names = {
        spec.name
        for spec in config.tool_specs
        if spec.enabled
    }

    generated_names = [
        tool.name
        for tool in generated_tools
    ]

    duplicate_generated_names = {
        name
        for name in generated_names
        if generated_names.count(name) > 1
    }

    if duplicate_generated_names:
        raise ValueError(
            "duplicate generated tool names: "
            + ", ".join(
                sorted(
                    duplicate_generated_names
                )
            )
        )

    conflicts = (
        builtin_names
        & set(generated_names)
    )

    if conflicts:
        raise ValueError(
            "generated tool names conflict "
            "with built-in tools: "
            + ", ".join(
                sorted(conflicts)
            )
        )


def build_policy_from_config(
    config: AgentConfig,
    *,
    transform_registry: TransformRegistry | None = None,
    runtime: CodeRuntime | None = None,
) -> ConfigurableAgentPolicy:

    return ConfigurableAgentPolicy(
        system_prompt=config.system_prompt,
        context_contract=config.context_contract,
        observation_contract=config.observation_contract,
        transform_registry=transform_registry,
        runtime=runtime,
    )


_HTTP_CLIENT: httpx.Client | None = None
_OPENROUTER_CLIENT: OpenRouter | None = None


def _http_client() -> httpx.Client:

    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        proxy = (
            os.environ.get("ALL_PROXY")
            or os.environ.get("HTTPS_PROXY")
        )

        _HTTP_CLIENT = httpx.Client(
            proxy=proxy,
            follow_redirects=True,
            timeout=httpx.Timeout(
                connect=15.0,
                read=180.0,
                write=30.0,
                pool=15.0
            ),
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=30.0
            ),
        )

    return _HTTP_CLIENT


def _openrouter_client() -> OpenRouter:

    global _OPENROUTER_CLIENT

    if _OPENROUTER_CLIENT is None:
        api_key = os.getenv("AGENT_API_KEY")

        if not api_key:
            raise ValueError("AGENT_API_KEY is not set")

        retry_config = RetryConfig(
            strategy="backoff",
            backoff=BackoffStrategy(
                initial_interval=2_000,
                max_interval=60_000,
                exponent=2.0,
                max_elapsed_time=180_000,
                jitter_ms=1_000
            ),
            retry_connection_errors=True,
            status_codes_override=[
                "408", "429", "500", "502", "503", "504"
            ]
        )

        _OPENROUTER_CLIENT = OpenRouter(
            api_key=api_key,
            client=_http_client(),
            retry_config=retry_config,
            timeout_ms=180_000,
        )

    return _OPENROUTER_CLIENT


def _provider_routing(config: AgentConfig) -> dict[str, Any]:
    routing = asdict(config.provider_routing)
    routing["order"] = list(config.provider_routing.order)
    return routing


def _resoning_config(config: AgentConfig) -> dict[str, Any]:
    return {
        key: value
        for key, value in asdict(config.reasoning).items()
        if value is not None
    }


def build_model_from_config(config: AgentConfig) -> BaseChatModel:

    api_key = os.getenv("AGENT_API_KEY")

    if not api_key:
        raise ValueError("AGENT_API_KEY is not set")

    forbidden_model_kwargs = {
        "model",
        "client",
        "api_key",
        "openrouter_api_key",
        "base_url",
        "max_retries",
        "timeout",
        "reasoning",
        "provider",
        "openrouter_provider"
    }

    conflicts = forbidden_model_kwargs & config.model_kwargs.keys()

    if conflicts:
        raise ValueError("reserved keys in model_kwargs: " + ", ".join(sorted(conflicts)))

    return ChatOpenRouter(
        model=config.model,
        api_key=api_key,
        client=_openrouter_client(),
        openrouter_provider=_provider_routing(config),
        reasoning=_resoning_config(config),
        max_retries=0,
        **config.model_kwargs,
    )


def resolve_agent_generated_paths(
    *,
    generated_root: str | Path | None = None,
    transform_directory: str | Path | None = None,
    generated_tool_directory: str | Path | None = None,
    verified_path: str | Path | None = None,
) -> GeneratedPaths:
    """Разрешает пути generated для сборки агента.

    Args:
        generated_root: Корень ``generated/``.
        transform_directory: Override каталога transforms.
        generated_tool_directory: Override каталога tools.
        verified_path: Override пути ``verified.json``.

    Returns:
        Нормализованные пути.
    """
    return resolve_generated_paths(
        generated_root,
        tools_dir=generated_tool_directory,
        transforms_dir=transform_directory,
        verified_path=verified_path,
    )


def build_agent_from_config(
    config: AgentConfig,
    *,
    model_factory: ModelFactory | None = None,
    generated_root: str | Path | None = None,
    transform_directory: str | Path | None = None,
    generated_tool_directory: str | Path | None = None,
    verified_path: str | Path | None = None,
) -> ReActAgent:
    """Собирает ``ReActAgent`` из конфигурации.

    Args:
        config: Конфигурация агента.
        model_factory: Фабрика LLM; по умолчанию OpenRouter.
        generated_root: Корень generated соседнего проекта.
        transform_directory: Override каталога transforms.
        generated_tool_directory: Override каталога tools.
        verified_path: Override пути ``verified.json``.
    """
    paths = resolve_agent_generated_paths(
        generated_root=generated_root,
        transform_directory=transform_directory,
        generated_tool_directory=generated_tool_directory,
        verified_path=verified_path,
    )

    if model_factory is None:
        model = build_model_from_config(config)
    else:
        model = model_factory(config)

    runtime = LocalRuntime()

    try:
        transform_registry = build_transform_registry(
            config,
            runtime=runtime,
            transform_directory=paths.transforms_dir,
            verified_path=paths.verified_path,
        )

        generated_tool_registry = build_generated_tool_registry(
            config,
            runtime=runtime,
            generated_tool_directory=paths.tools_dir,
            verified_path=paths.verified_path,
        )

        generated_tools = build_generated_tools(
            config,
            registry=generated_tool_registry,
        )

        validate_tool_name_conflicts(
            config,
            generated_tools,
        )

        policy = build_policy_from_config(
            config,
            transform_registry=transform_registry,
            runtime=runtime,
        )

        metadata = build_config_metadata(
            config,
            transform_registry=transform_registry,
            generated_tool_registry=generated_tool_registry,
        )

        return ReActAgent(
            model=model,
            policy=policy,
            recursion_limit=config.recursion_limit,
            initial_user_message=config.initial_user_message,
            tool_options=build_tool_options(config),
            additional_tools=generated_tools,
            config_id=config.config_id,
            metadata=metadata,
        )

    except Exception:
        runtime.close()
        raise
