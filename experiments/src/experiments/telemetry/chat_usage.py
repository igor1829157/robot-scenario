import argparse
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


class ChatUsageError(RuntimeError):
    """Ошибка чтения или разбора chat.jsonl"""


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    
    if isinstance(value, int):
        return max(value, 0)

    return 0


def _decimal(value: Any) -> Decimal:
    if value is None or isinstance(value, bool):
        return Decimal("0")

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal("0")


def extract_chat_usage(
    chat_log_path: str | Path,
) -> dict[str, Any]:
    
    path = Path(chat_log_path)

    if not path.is_file():
        raise ChatUsageError(f"chat log does not exist: {path}")

    generation_ids: list[str] = []
    seen_generation_ids: set[str] = set()

    reported_models: set[str] = set()
    langchain_providers: set[str] = set()

    calls = 0
    calls_without_generation_id = 0
    calls_with_reasoning = 0

    input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0
    cache_read_tokens = 0
    cache_creation_tokens = 0

    max_input_tokens_per_call = 0
    max_output_tokens_per_call = 0
    max_reasoning_tokens_per_call = 0

    inline_cost = Decimal("0")
    malformed_lines: list[int] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines.append(line_number)
                continue

            if record.get("kind") != "message":
                continue

            message = record.get("data")

            if not isinstance(message, dict):
                continue

            if message.get("type") != "ai":
                continue

            response_metadata = (
                message.get("response_metadata") or {}
            )
            usage_metadata = (
                message.get("usage_metadata") or {}
            )
            additional_kwargs = (
                message.get("additional_kwargs") or {}
            )

            generation_id = response_metadata.get("id")

            if isinstance(generation_id, str) and generation_id:
                if generation_id in seen_generation_ids:
                    continue

                seen_generation_ids.add(generation_id)
                generation_ids.append(generation_id)
            else:
                calls_without_generation_id += 1

            calls += 1

            model_name = response_metadata.get("model_name")
            if isinstance(model_name, str) and model_name:
                reported_models.add(model_name)

            provider = response_metadata.get("model_provider")
            if isinstance(provider, str) and provider:
                langchain_providers.add(provider)

            call_input_tokens = _non_negative_int(
                usage_metadata.get("input_tokens")
            )
            call_output_tokens = _non_negative_int(
                usage_metadata.get("output_tokens")
            )

            input_details = (
                usage_metadata.get("input_token_details") or {}
            )
            output_details = (
                usage_metadata.get("output_token_details") or {}
            )

            call_reasoning_tokens = _non_negative_int(
                output_details.get("reasoning")
            )
            call_cache_read_tokens = _non_negative_int(
                input_details.get("cache_read")
            )
            call_cache_creation_tokens = _non_negative_int(
                input_details.get("cache_creation")
            )

            input_tokens += call_input_tokens
            output_tokens += call_output_tokens
            reasoning_tokens += call_reasoning_tokens
            cache_read_tokens += call_cache_read_tokens
            cache_creation_tokens += call_cache_creation_tokens

            max_input_tokens_per_call = max(
                max_input_tokens_per_call,
                call_input_tokens,
            )
            max_output_tokens_per_call = max(
                max_output_tokens_per_call,
                call_output_tokens,
            )
            max_reasoning_tokens_per_call = max(
                max_reasoning_tokens_per_call,
                call_reasoning_tokens,
            )

            inline_cost += _decimal(
                response_metadata.get("cost")
            )

            reasoning_content = additional_kwargs.get(
                "reasoning_content"
            )
            reasoning_details = additional_kwargs.get(
                "reasoning_details"
            )

            if reasoning_content or reasoning_details:
                calls_with_reasoning += 1

    if malformed_lines:
        status = "partial"
    elif calls == 0:
        status = "empty"
    else:
        status = "inline_only"

    non_reasoning_output_tokens = max(0, output_tokens - reasoning_tokens)

    if output_tokens:
        reasoning_share = Decimal(reasoning_tokens) / Decimal(output_tokens)
    else:
        reasoning_share = Decimal("0")

    return {
        "status": status,
        "calls": calls,
        "calls_with_generation_id": len(generation_ids),
        "calls_without_generation_id": (
            calls_without_generation_id
        ),
        "generation_ids": generation_ids,
        "reported_models": sorted(reported_models),
        "langchain_providers": sorted(
            langchain_providers
        ),
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "non_reasoning_output_tokens": (
                non_reasoning_output_tokens
            ),
            "cache_read_tokens": cache_read_tokens,
            "cache_creation_tokens": (
                cache_creation_tokens
            ),
            "reasoning_share_of_output": format(
                reasoning_share,
                ".6f",
            ),
            "max_input_tokens_per_call": (
                max_input_tokens_per_call
            ),
            "max_output_tokens_per_call": (
                max_output_tokens_per_call
            ),
            "max_reasoning_tokens_per_call": (
                max_reasoning_tokens_per_call
            ),
            "inline_cost_usd": format(
                inline_cost,
                "f",
            ),
            "currency": "USD",
        },
        "reasoning_output": {
            "calls_with_reasoning": (
                calls_with_reasoning
            ),
            "available_for_all_calls": (
                calls > 0
                and calls_with_reasoning == calls
            ),
        },
        "integrity": {
            "malformed_line_count": len(
                malformed_lines
            ),
            "malformed_lines": malformed_lines,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract LLM usage from chat.jsonl"
    )
    parser.add_argument("chat_log")
    parser.add_argument("--output")
    args = parser.parse_args()

    result = extract_chat_usage(args.chat_log)

    serialized = json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        tmp_path = output_path.with_name(
            output_path.name + ".tmp"
        )
        tmp_path.write_text(
            serialized + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(output_path)

        print(output_path)
    else:
        print(serialized)


if __name__ == "__main__":
    main()
