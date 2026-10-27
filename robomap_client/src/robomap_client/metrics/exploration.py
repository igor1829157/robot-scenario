from typing import Any


def calculate_exploration_metrics(
    full_state: dict[str, Any],
    robot_id: str = "robot_1"
) -> dict[str, Any]:

    real_map = full_state.get("real_map", [])
    robots_memory = full_state.get("robots_memory", {}).get(robot_id, [])

    target_real_objects = [
        obj for obj in real_map
        if obj.get("object") != "robot" and not obj.get("location_id")
    ]
    total_targets = len(target_real_objects)

    mem_lookup = {}
    for mem in robots_memory:
        if not mem.get("is_virtual") and not mem.get("location_id"):
            real_id = mem.get("real_id")
            mem_lookup[real_id] = mem
    
    discovered_count = 0
    fully_explored_count = 0

    for real_obj in target_real_objects:
        real_id = real_obj.get("id")
        if real_id in mem_lookup:
            discovered_count += 1
            if mem_lookup[real_id].get("is_fully_explored", False):
                fully_explored_count += 1
    
    discovery_rate = (
        discovered_count / total_targets * 100 if total_targets > 0 else 0
    )

    explore_rate = (
        fully_explored_count / total_targets * 100 if total_targets > 0 else 0
    )

    return {
        "total_top_level_objects": total_targets,
        "discovered_objects": discovered_count,
        "fully_explored_objects": fully_explored_count,
        "discovery_percentage": round(discovery_rate, 2),
        "fully_explored_percentage": round(explore_rate, 2),
        "final_tick": full_state.get("tick", 0),
    }
