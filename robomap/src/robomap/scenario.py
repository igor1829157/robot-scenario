import os
import importlib.util
from robomap.api import API
import logging

logger = logging.getLogger(__name__)

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

class BaseScenario:
    """Базовый интерфейс для автодополнения. Можно наследовать, но не обязательно."""
    def __init__(self, api: API, auto_robot: bool = True):
        self.api = api
        self.world = api.world
        self.auto_robot = auto_robot
        self.stage = "peace"
        self.terminal_stages = ["completed", "failed"]
    def setup(self): pass
    def tick(self, world): pass

def load_scenario(scenario_name: str, api: API, auto_robot: bool):
    """
    Ищет и загружает скрипт сценария из data/scenarios/{name}/script.py.
    Если файла нет или произошла ошибка — возвращает None (Режим Песочницы).
    """
    if scenario_name in ["empty", "sandbox", "none", ""]:
        return None
        
    script_path = os.path.join(project_path, "data", "scenarios", scenario_name, "script.py")
    
    if not os.path.exists(script_path):
        logger.warning("SCENARIO LOADER | Сценарий '%s' не найден по пути %s. Запуск Песочницы.", scenario_name, script_path)
        return None
        
    try:
        spec = importlib.util.spec_from_file_location(f"dynamic_scenario_{scenario_name}", script_path)
        if spec is None or spec.loader is None:
            logger.error("SCENARIO LOADER | Не удалось создать спецификацию модуля для '%s'.", scenario_name)
            return None
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'Scenario'):
            scenario_instance = module.Scenario(api, auto_robot=auto_robot)
            logger.info("SCENARIO LOADER | Успешно загружен плагин: %s", scenario_name)
            return scenario_instance
        else:
            logger.error("SCENARIO LOADER | В файле %s отсутствует класс 'Scenario'. Запуск Песочницы.", script_path)
            return None
            
    except Exception as e:
        logger.error("SCENARIO LOADER | Критическая ошибка при инициализации '%s': %s. Запуск Песочницы.", scenario_name, str(e))
        return None