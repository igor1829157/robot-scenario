import argparse
import json
import os
import random
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import httpx

from experiments.telemetry.chat_usage import (
    extract_chat_usage,
)


_GENERATION_URL = (
    "https://openrouter.ai/api/v1/generation"
)

_RETRYABLE_STATUS_CODES = {
    408,
    429,
    500,
    502,
    503,
    504,
}


class OpenRouterMetadataError(RuntimeError):
    """Ошибка получения OpenRouter metadata."""


def _decimal(value: Any) -> Decimal:
    if value is None or isinstance(value, bool):
        return Decimal("0")

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal("0")


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0

    if isinstance(value, int):
        return max(value, 0)

    return 0


def _retry_delay(
    response: httpx.Response,
    *,
    attempt: int,
) -> float:
    retry_after = response.headers.get("Retry-After")

    if retry_after is not None:
        try:
            return max(float(retry_after), 0.0)
        except ValueError:
            pass

    exponential = min(
        2 ** (attempt - 1),
        30,
    )
    return exponential + random.uniform(0.0, 1.0)


def _create_http_client() -> httpx.Client:
    proxy = (
        os.environ.get("ALL_PROXY")
        or os.environ.get("HTTPS_PROXY")
    )

    return httpx.Client(
        proxy=proxy,
        follow_redirects=True,
        timeout=httpx.Timeout(
            connect=15.0,
            read=60.0,
            write=30.0,
            pool=15.0,
        ),
        limits=httpx.Limits(
            max_connections=5,
            max_keepalive_connections=5,
            keepalive_expiry=30.0,
        ),
    )


def fetch_generation(
    client: httpx.Client,
    *,
    api_key: str,
    generation_id: str,
    max_attempts: int = 5,
) -> dict[str, Any]:
    """Получает metadata одной генерации."""

    last_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.get(
                _GENERATION_URL,
                params={"id": generation_id},
                headers={
                    "Authorization": f"Bearer {api_key}",
                },
            )
        except httpx.HTTPError as exc:
            last_error = (
                f"{type(exc).__name__}: {exc}"
            )

            if attempt == max_attempts:
                break

            delay = min(
                2 ** (attempt - 1),
                30,
            )
            delay += random.uniform(0.0, 1.0)
            time.sleep(delay)
            continue

        if response.status_code == 200:
            payload = response.json()
            data = payload.get("data")

            if not isinstance(data, dict):
                raise OpenRouterMetadataError(
                    "generation response has no data object: "
                    f"{generation_id}"
                )

            return data

        last_error = (
            f"HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )

        if (
            response.status_code
            not in _RETRYABLE_STATUS_CODES
            or attempt == max_attempts
        ):
            break

        time.sleep(
            _retry_delay(
                response,
                attempt=attempt,
            )
        )

    raise OpenRouterMetadataError(
        f"failed to fetch generation "
        f"{generation_id}: {last_error}"
    )


def _normalize_generation(
    data: dict[str, Any],
) -> dict[str, Any]:
    provider_responses: list[dict[str, Any]] = []

    for response in data.get("provider_responses") or []:
        if not isinstance(response, dict):
            continue

        provider_responses.append({
            "endpoint_id": response.get("endpoint_id"),
            "upstream_id": response.get("id"),
            "model_permaslug": response.get(
                "model_permaslug"
            ),
            "provider_name": response.get(
                "provider_name"
            ),
            "status": response.get("status"),
            "latency_ms": response.get("latency"),
            "is_byok": response.get("is_byok"),
        })

    return {
        "id": data.get("id"),
        "request_id": data.get("request_id"),
        "upstream_id": data.get("upstream_id"),
        "created_at": data.get("created_at"),
        "model": data.get("model"),
        "provider_name": data.get("provider_name"),
        "provider_responses": provider_responses,
        "streamed": data.get("streamed"),
        "cancelled": data.get("cancelled"),
        "finish_reason": data.get("finish_reason"),
        "native_finish_reason": data.get(
            "native_finish_reason"
        ),
        "native_tokens_prompt": data.get(
            "native_tokens_prompt"
        ),
        "native_tokens_completion": data.get(
            "native_tokens_completion"
        ),
        "native_tokens_reasoning": data.get(
            "native_tokens_reasoning"
        ),
        "native_tokens_cached": data.get(
            "native_tokens_cached"
        ),
        "total_cost_usd": (
            str(data["total_cost"])
            if data.get("total_cost") is not None
            else None
        ),
        "latency_ms": data.get("latency"),
        "generation_time_ms": data.get(
            "generation_time"
        ),
        "data_region": data.get("data_region"),
        "is_byok": data.get("is_byok"),
    }


def collect_openrouter_metadata(
    chat_log_path: str | Path,
) -> dict[str, Any]:
    """Обогащает chat usage через OpenRouter API."""

    api_key = os.getenv("AGENT_API_KEY")

    if not api_key:
        raise OpenRouterMetadataError(
            "AGENT_API_KEY is not set"
        )

    inline = extract_chat_usage(chat_log_path)
    generation_ids = inline["generation_ids"]

    generations: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    with _create_http_client() as client:
        for generation_id in generation_ids:
            try:
                data = fetch_generation(
                    client,
                    api_key=api_key,
                    generation_id=generation_id,
                )
                generations.append(
                    _normalize_generation(data)
                )
            except Exception as exc:
                errors.append({
                    "generation_id": generation_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                })

    models: set[str] = set()
    model_permaslugs: set[str] = set()
    upstream_providers: set[str] = set()
    endpoint_ids: set[str] = set()

    native_prompt_tokens = 0
    native_completion_tokens = 0
    native_reasoning_tokens = 0
    native_cached_tokens = 0
    verified_cost = Decimal("0")

    for generation in generations:
        model = generation.get("model")
        if isinstance(model, str) and model:
            models.add(model)

        provider = generation.get("provider_name")
        if isinstance(provider, str) and provider:
            upstream_providers.add(provider)

        native_prompt_tokens += _non_negative_int(
            generation.get("native_tokens_prompt")
        )
        native_completion_tokens += _non_negative_int(
            generation.get("native_tokens_completion")
        )
        native_reasoning_tokens += _non_negative_int(
            generation.get("native_tokens_reasoning")
        )
        native_cached_tokens += _non_negative_int(
            generation.get("native_tokens_cached")
        )

        verified_cost += _decimal(
            generation.get("total_cost_usd")
        )

        for provider_response in (
            generation.get("provider_responses") or []
        ):
            permaslug = provider_response.get(
                "model_permaslug"
            )
            if isinstance(permaslug, str) and permaslug:
                model_permaslugs.add(permaslug)

            endpoint_id = provider_response.get(
                "endpoint_id"
            )
            if isinstance(endpoint_id, str) and endpoint_id:
                endpoint_ids.add(endpoint_id)

            response_provider = provider_response.get(
                "provider_name"
            )
            if (
                isinstance(response_provider, str)
                and response_provider
            ):
                upstream_providers.add(
                    response_provider
                )

    if not generation_ids:
        status = "empty"
    elif errors:
        status = "partial"
    else:
        status = "verified"

    return {
        "schema_version": 1,
        "status": status,
        "source_chat_log": str(chat_log_path),
        "inline": inline,
        "summary": {
            "requested_generations": len(
                generation_ids
            ),
            "resolved_generations": len(
                generations
            ),
            "failed_generations": len(errors),
            "models": sorted(models),
            "model_permaslugs": sorted(
                model_permaslugs
            ),
            "upstream_providers": sorted(
                upstream_providers
            ),
            "endpoint_ids": sorted(endpoint_ids),
            "native_prompt_tokens": (
                native_prompt_tokens
            ),
            "native_completion_tokens": (
                native_completion_tokens
            ),
            "native_reasoning_tokens": (
                native_reasoning_tokens
            ),
            "native_cached_tokens": (
                native_cached_tokens
            ),
            "verified_cost_usd": format(
                verified_cost,
                "f",
            ),
            "currency": "USD",
        },
        "generations": generations,
        "errors": errors,
    }


def _write_json_atomic(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    tmp_path = path.with_name(
        path.name + ".tmp"
    )
    tmp_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect OpenRouter generation metadata"
    )
    parser.add_argument("chat_log")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = collect_openrouter_metadata(
        args.chat_log
    )

    output_path = Path(args.output)
    _write_json_atomic(output_path, result)
    print(output_path)


if __name__ == "__main__":
    main()