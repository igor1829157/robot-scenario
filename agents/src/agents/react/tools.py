from typing import Any

from langchain_core.tools import BaseTool, tool
from robomap_client.core.client import RobomapClient

from agents.react.policy import AgentPolicy


_TOOL_NAMES = {
    "move",
    "rotate",
    "finish_exploration",
}


def create_robot_tools(
    client: RobomapClient,
    policy: AgentPolicy,
    *,
    tool_options: dict[str, dict[str, Any]] | None = None,
) -> list[BaseTool]:
    """Создаёт настраиваемые инструменты робота."""

    options = tool_options or {}

    unknown_tools = set(options) - _TOOL_NAMES
    if unknown_tools:
        names = ", ".join(sorted(unknown_tools))
        raise ValueError(f"unknown tools: {names}")

    move_options = options.get("move", {})
    rotate_options = options.get("rotate", {})
    finish_options = options.get("finish_exploration", {})

    @tool
    def move(target_x: float, target_y: float) -> str:
        """Move toward absolute map coordinates.

        Blocks until the move finishes, fails, or times out.
        """
        commands_kwargs = dict(
            move_options.get("kwargs_defaults") or {}
        )

        commands_kwargs.update(
            {
                "target_x": round(float(target_x), 2),
                "target_y": round(float(target_y), 2),
            }
        )

        max_wait_ticks = move_options.get("max_wait_ticks", 200)

        result = client.execute_and_wait(
            "move",
            max_wait_ticks=max_wait_ticks,
            **commands_kwargs,
        )
        return policy.format_observation(result)

    @tool
    def rotate(target_angle: float) -> str:
        """Rotate in place to an absolute world heading in degrees.

        0 = East (+X), 90 = North (+Y), 180 = West (-X), 270 = South (-Y).
        """
        commands_kwargs = dict(
            mode="absolute",
        )

        commands_kwargs.update(
            rotate_options.get("kwargs_defaults") or {}
        )

        commands_kwargs["target"]= round(float(target_angle) % 360.0, 2)

        max_wait_ticks = rotate_options.get("max_wait_ticks", 200)

        result = client.execute_and_wait(
            "rotate",
            max_wait_ticks=max_wait_ticks,
            **commands_kwargs,
        )
        return policy.format_observation(result)

    @tool
    def finish_exploration() -> str:
        """Call this when the map is reasonably explored and the mission should end."""
        return policy.format_observation({"status": "COMPLETED"})


    available_tools: dict[str, BaseTool] = {
        "move": move,
        "rotate": rotate,
        "finish_exploration": finish_exploration,
    }

    result_tools: list[BaseTool] = []
    for tool_name, robot_tool in available_tools.items():
        tool_config = options.get(tool_name, {})

        if not tool_config.get("enabled", True):
            continue

        custom_description = tool_config.get("description")
        if custom_description:
            robot_tool.description = custom_description

        result_tools.append(robot_tool)

    return result_tools
