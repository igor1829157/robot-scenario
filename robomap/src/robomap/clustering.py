import math
import copy
import logging

logger = logging.getLogger(__name__)


class IncrementalObjectClusterer:
    """
    Алгоритм инкрементальной кластеризации объектов в 2D пространстве.
    Группирует объекты в памяти робота на основе расстояния eps.
    """
    def __init__(self, eps: float, min_samples_to_form_cluster: int = 1):
        self.eps: float = eps
        self.min_samples_to_form_cluster: int = min_samples_to_form_cluster
        self.reset()

    def reset(self):
        self.clusters: list[dict] = []
        self.next_cluster_id: int = 0
        self.robot_memory: dict[str, dict] = {}

    def get_state(self) -> dict:
        serializable_clusters = []
        for cluster in self.clusters:
            c_copy = cluster.copy()
            c_copy['object_ids'] = sorted(list(c_copy['object_ids']))
            serializable_clusters.append(c_copy)

        return {
            "clusters": serializable_clusters,
            "next_cluster_id": self.next_cluster_id
        }

    def set_state(self, state: dict, full_robot_memory_list: list[dict]):
        self.next_cluster_id = state.get("next_cluster_id", 0)

        loaded_clusters = state.get("clusters", [])
        self.clusters = []
        for cluster_data in loaded_clusters:
            c_copy = cluster_data.copy()
            c_copy['object_ids'] = set(c_copy.get('object_ids', []))
            self.clusters.append(c_copy)

        self.robot_memory.clear()
        for obj_data in full_robot_memory_list:
            if 'id' in obj_data:
                self.robot_memory[str(obj_data['id'])] = obj_data

        logger.info("Clusterer state restored: %d clusters, next ID is %d.", len(self.clusters), self.next_cluster_id)

    def _distance(self, obj1_coords: tuple[float, float], obj2_coords: tuple[float, float]) -> float:
        return math.hypot(obj1_coords[0] - obj2_coords[0], obj1_coords[1] - obj2_coords[1])

    def _get_object_coords(self, obj_id: str) -> tuple[float, float] | None:
        if obj_id in self.robot_memory:
            obj = self.robot_memory[obj_id]
            try:
                if obj.get('x') is not None and obj.get('y') is not None:
                    return (float(obj['x']), float(obj['y']))
            except (ValueError, TypeError):
                return None
        return None

    def _find_cluster_by_object_id(self, obj_id: str) -> dict | None:
        for cluster in self.clusters:
            if obj_id in cluster['object_ids']:
                return cluster
        return None

    def _update_cluster_name(self, cluster: dict):
        if not cluster or not cluster.get('object_ids'):
            cluster['name'] = "empty cluster"
            return

        objects_in_cluster = [self.robot_memory[obj_id] for obj_id in cluster['object_ids'] if obj_id in self.robot_memory]

        if not objects_in_cluster:
            cluster['name'] = "Cluster with disappearing objects"
            return

        cluster['name'] = f"Cluster {cluster['id']}"

    def _recalculate_cluster_connectivity(self, cluster_id_to_check: int):
        cluster = next((c for c in self.clusters if c['id'] == cluster_id_to_check), None)
        if not cluster or not cluster['object_ids'] or len(cluster['object_ids']) <= 1:
            return

        current_cluster_objects = list(cluster['object_ids'])
        visited_obj_ids = set()
        new_sub_clusters_obj_ids = []

        for obj_id_start in current_cluster_objects:
            if obj_id_start not in visited_obj_ids:
                current_component_obj_ids = set()
                q = [obj_id_start]
                visited_obj_ids.add(obj_id_start)
                current_component_obj_ids.add(obj_id_start)

                head = 0
                while head < len(q):
                    curr_obj_id = q[head]
                    head += 1
                    curr_coords = self._get_object_coords(curr_obj_id)
                    if not curr_coords: 
                        continue

                    for other_obj_id in current_cluster_objects:
                        if other_obj_id not in visited_obj_ids:
                            other_coords = self._get_object_coords(other_obj_id)
                            if not other_coords: 
                                continue

                            if self._distance(curr_coords, other_coords) <= self.eps:
                                visited_obj_ids.add(other_obj_id)
                                current_component_obj_ids.add(other_obj_id)
                                q.append(other_obj_id)
                new_sub_clusters_obj_ids.append(current_component_obj_ids)

        if len(new_sub_clusters_obj_ids) > 1:
            logger.info("Cluster %d split into %d sub-clusters.", cluster['id'], len(new_sub_clusters_obj_ids))
            self.clusters = [c for c in self.clusters if c['id'] != cluster_id_to_check]

            newly_created_clusters = []
            for sub_cluster_obj_ids in new_sub_clusters_obj_ids:
                if sub_cluster_obj_ids:
                    new_id = self.next_cluster_id
                    self.next_cluster_id += 1
                    new_cluster = {'id': new_id, 'object_ids': sub_cluster_obj_ids, 'name': ''}
                    newly_created_clusters.append(new_cluster)

            self.clusters.extend(newly_created_clusters)

            for new_cluster in newly_created_clusters:
                self._update_cluster_name(new_cluster)
                logger.info("  New sub-cluster %d ('%s') created.", new_cluster['id'], new_cluster['name'])

    def update_clusters(self, current_step_data: dict) -> list[dict]:
        newly_detected = current_step_data['newly detected objects']
        disappeared_objects = current_step_data['disappeared objects']
        full_robot_memory_list = current_step_data['full robot memory']

        current_memory_ids = set()
        for obj_data in full_robot_memory_list:
            if 'id' in obj_data:
                obj_id_str = str(obj_data['id'])
                self.robot_memory[obj_id_str] = obj_data
                current_memory_ids.add(obj_id_str)

        for obj_id in list(self.robot_memory.keys()):
            if obj_id not in current_memory_ids:
                del self.robot_memory[obj_id]

        clusters_to_update_name = set()
        original_clusters_to_check_for_split = set()

        for obj_data in disappeared_objects:
            if 'id' not in obj_data:
                continue
            obj_id = str(obj_data['id'])
            cluster_obj_was_in = self._find_cluster_by_object_id(obj_id)
            if cluster_obj_was_in:
                cluster_obj_was_in['object_ids'].remove(obj_id)
                if not cluster_obj_was_in['object_ids']:
                    self.clusters = [c for c in self.clusters if c['id'] != cluster_obj_was_in['id']]
                else:
                    clusters_to_update_name.add(cluster_obj_was_in['id'])
                    original_clusters_to_check_for_split.add(cluster_obj_was_in['id'])

        for cluster_id in clusters_to_update_name:
            cluster = next((c for c in self.clusters if c['id'] == cluster_id), None)
            if cluster:
                self._update_cluster_name(cluster)

        for cluster_id_to_check in original_clusters_to_check_for_split:
            self._recalculate_cluster_connectivity(cluster_id_to_check)

        clusters_to_update_name.clear()

        for new_obj_data in newly_detected:
            if 'id' not in new_obj_data:
                continue
            new_obj_id = str(new_obj_data['id'])
            new_obj_coords = self._get_object_coords(new_obj_id)
            if not new_obj_coords: 
                continue

            if self._find_cluster_by_object_id(new_obj_id):
                continue

            candidate_clusters_to_join = []
            for cluster in self.clusters:
                for existing_obj_id in cluster['object_ids']:
                    existing_obj_coords = self._get_object_coords(existing_obj_id)
                    if existing_obj_coords and self._distance(new_obj_coords, existing_obj_coords) <= self.eps:
                        candidate_clusters_to_join.append(cluster)
                        break

            if not candidate_clusters_to_join:
                new_cluster_id = self.next_cluster_id
                self.next_cluster_id += 1
                new_cluster = {'id': new_cluster_id, 'object_ids': {new_obj_id}, 'name': ''}
                self.clusters.append(new_cluster)
                self._update_cluster_name(new_cluster)

            elif len(candidate_clusters_to_join) == 1:
                target_cluster = candidate_clusters_to_join[0]
                target_cluster['object_ids'].add(new_obj_id)
                self._update_cluster_name(target_cluster)

            else:
                target_cluster = candidate_clusters_to_join[0]
                target_cluster['object_ids'].add(new_obj_id)

                merged_ids = {c['id'] for c in candidate_clusters_to_join[1:]}

                for cluster_to_merge in candidate_clusters_to_join[1:]:
                    if cluster_to_merge['id'] == target_cluster['id']: 
                        continue
                    target_cluster['object_ids'].update(cluster_to_merge['object_ids'])

                self.clusters = [c for c in self.clusters if c['id'] not in merged_ids]
                self._update_cluster_name(target_cluster)

        return copy.deepcopy(self.clusters)

    def get_clusters_with_data(self) -> list[dict]:
        output_clusters = []
        for c_internal in self.clusters:
            object_types_in_cluster = set()
            for obj_id in c_internal['object_ids']:
                if obj_id in self.robot_memory:
                    obj_info = self.robot_memory[obj_id]
                    obj_name = obj_info.get('object', obj_info.get('name', 'unknown'))
                    object_types_in_cluster.add(obj_name)

            output_clusters.append({
                'id': c_internal['id'],
                'name': c_internal.get('name', f"Cluster {c_internal['id']}"),
                'object_ids': list(c_internal['object_ids']),
                'object_types': sorted(list(object_types_in_cluster)),
                'num_objects': len(c_internal['object_ids'])
            })
        return output_clusters