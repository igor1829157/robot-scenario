import math
from robomap.api import API

def distance(obj1, obj2):
    if getattr(obj1, 'x', None) is None or getattr(obj2, 'x', None) is None: 
        return 999.0
    return math.hypot(obj1.x - obj2.x, obj1.y - obj2.y)

class Scenario:
    def __init__(self, api: API, auto_robot: bool = True):
        self.api = api
        self.world = api.world
        self.auto_robot = auto_robot
        self.stage = "peace_1"
        self.terminal_stages = ["bear_fleeing", "failed"]
        self.auto_step = 0
        self.bear, self.tent, self.human, self.flare = None, None, None, None
        self.flare_id = "flare_1"
        self.robot_base = (63, -38)
        self.flare_pos = (63, -46)
        self.bear_start = (50, -5)
        self.intercept_pos = (50, -38) # Робот ждет у палатки
        self.detour_pos = None # Точка, куда отступает медведь

    def _assign_patrol(self):
        """Человек гуляет между палаткой и медведем"""
        self.api.command_action(self.human.id, "clear")
        self.api.command_move(self.human.id, 50, -30)
        self.api.command_move(self.human.id, 50, -15, queue=True)

    def setup(self):
        print("\n🎬 ЗАПУСК СЦЕНАРИЯ: Нападение Медведя (Фикс таймингов)!")
        with self.api.lock:
            # Увеличили прочность палатки, чтобы она не ломалась за 0.5 сек
            self.tent = self.world.add_object("tent", 50, -40, obj_id="tent_1")
            self.tent.dynamic = False; self.tent.affordance = ["enter", "break"]; self.tent.durability = 100.0

            self.human = self.world.add_object("human", 50, -30, obj_id="human_1")
            self.human.dynamic = True; self.human.behavior = ["move", "speak"]
            self.human.vision_radius = 20.0; self.human.speed = 1.5
            self.human.initialize_physics()
            
            self.bear = self.world.add_object("bear", self.bear_start[0], self.bear_start[1], obj_id="bear_1")
            self.bear.dynamic = True; self.bear.behavior = ["move", "break", "speak"]
            self.bear.damage = 10.0; self.bear.speed = 0.8
            self.bear.initialize_physics()
            
            self.flare = self.world.add_object("flare", self.flare_pos[0], self.flare_pos[1], obj_id=self.flare_id)
            self.flare.dynamic = False; self.flare.affordance = ["pickup"]

            robot = self.world.robots.get("robot_1")
            if not robot:
                from robomap.models import Robot
                eps = self.world.config.get("clustering", {}).get("eps", 0.0)
                robot = Robot(obj_id="robot_1", x=self.robot_base[0], y=self.robot_base[1], vision_radius=15.0, fov=360.0, eps=eps)
                self.world.robots["robot_1"] = robot
            robot.speed = 3.5 
            robot.fov = 360.0
            robot.initialize_physics()

            self._assign_patrol()
            
            self.api.command_move(self.bear.id, self.tent.x, self.tent.y)


    def tick(self, world):
        if not self.bear or not self.tent or not self.human: return
        robot = world.robots.get("robot_1")

        # Патруль человека (работает, пока все тихо)
        if self.stage in ["peace_1", "bear_retreating", "bear_returning"] and len(self.human.action_queue) == 0:
            self._assign_patrol()

        # =====================================================================
        # ГЛОБАЛЬНЫЕ ПРАВИЛА МИРА (Триггеры Сценария)
        # =====================================================================
        
        # ПРАВИЛО 1: ПОБЕДА (Шашка на земле рядом с медведем) - СРАБАТЫВАЕТ ВСЕГДА
        if self.stage != "bear_fleeing":
            flare_on_ground = next((obj for obj in world.real_map_data if obj.id == self.flare_id and obj.location_id is None), None)
            if flare_on_ground and distance(flare_on_ground, self.bear) < 10:
                self.stage = "bear_fleeing"
                print("\n🔥 ПРАВИЛО МИРА: Медведь увидел шашку! Он окончательно убегает!")
                self.api.command_action(self.bear.id, "clear")
                self.api.command_move(self.bear.id, 100, 100)
                self.api.command_action(self.human.id, "speak", "Ура! Сработало!", 5)
                return

        # ПРАВИЛО 2: ИСПУГ ОТ СЛОВ (Робот говорит "Брысь отсюда!") - СРАБАТЫВАЕТ В ЛЮБОЙ МОМЕНТ (кроме побега)
        if self.stage not in ["bear_fleeing", "bear_retreating"] and robot:
            robot_speaking = robot.state.get("speaking", "")
            if "Брысь" in robot_speaking and distance(robot, self.bear) < 10:
                self.stage = "bear_retreating"
                
                # Медведь делает огромный крюк влево (x-30) и вверх (y+20), чтобы точно не пересечься с патрулем
                self.detour_pos = (self.bear.x - 30, self.bear.y + 20)
                
                print(f"\n🔊 ПРАВИЛО МИРА: Медведь услышал Робота. Отбегает далеко в точку {self.detour_pos}!")
                self.api.command_action(self.bear.id, "clear")
                self.api.command_move(self.bear.id, self.detour_pos[0], self.detour_pos[1])
                
                self._assign_patrol()

        # ПРАВИЛО 2.5: Медведь завершил отступление и идет обратно к палатке
        if self.stage == "bear_retreating" and self.detour_pos:
            if math.hypot(self.bear.x - self.detour_pos[0], self.bear.y - self.detour_pos[1]) < 3.0:
                self.stage = "bear_returning"
                print("\n🐻 ПРАВИЛО МИРА: Медведь закончил маневр и снова двинулся к палатке!")
                self.api.command_action(self.bear.id, "clear")
                self.api.command_move(self.bear.id, self.tent.x, self.tent.y)

        # ПРАВИЛО 3: ЧЕЛОВЕК ЗАМЕЧАЕТ МЕДВЕДЯ
        if self.stage in ["peace_1", "bear_returning"]:
            # Увеличили радиус обнаружения до 16, чтобы было больше времени на реакцию
            if distance(self.human, self.bear) < 16.0:
                if self.stage == "peace_1":
                    self.stage = "bear_spotted_1"
                    print("\n😱 ПРАВИЛО МИРА: Человек увидел медведя и бежит к Роботу!")
                    self.api.command_action(self.human.id, "clear")
                    self.api.command_action(self.human.id, "speak", "ААА! Медведь!", 4)
                elif self.stage == "bear_returning":
                    self.stage = "bear_spotted_2"
                    print("\n😱 ПРАВИЛО МИРА: Человек увидел вернувшегося медведя и бежит к Роботу!")
                    self.api.command_action(self.human.id, "clear")
                    self.api.command_action(self.human.id, "speak", "ААА! Снова медведь!", 4)
                
                self.api.command_action(self.bear.id, "speak", "РРРР!", 3)
                if robot:
                    self.api.command_move(self.human.id, self.robot_base[0], self.robot_base[1])


        # =====================================================================
        # ДОПОЛНИТЕЛЬНАЯ ЛОГИКА МИРА: Медведь ломает палатку
        # =====================================================================
        if self.stage not in ["bear_fleeing", "bear_retreating", "failed"]:
            if distance(self.bear, self.tent) < 3.0 and len(self.bear.action_queue) == 0:
                self.api.command_action(self.bear.id, "break", self.tent.id)


        # =====================================================================
        # ЛОГИКА АВТО-РОБОТА (Имитация игрока)
        # =====================================================================
        if self.auto_robot and robot:
            
            # ЭТАП 0: Ждем пока человек добежит до базы в первый раз
            if self.auto_step == 0 and self.stage == "bear_spotted_1":
                if distance(self.human, robot) < 3.0:
                    self.api.command_action(robot.id, "clear")
                    self.api.command_move(robot.id, self.intercept_pos[0], self.intercept_pos[1])
                    self.auto_step = 1

            # ЭТАП 1: Робот пришел к палатке, ждет медведя на 9 метров
            elif self.auto_step == 1 and len(robot.action_queue) == 0:
                if distance(robot, self.bear) <= 9.0:
                    self.api.command_action(robot.id, "speak", "Брысь отсюда!", 3)
                    self.api.command_move(robot.id, self.robot_base[0], self.robot_base[1], queue=True)
                    self.auto_step = 2

            # ЭТАП 2: Реакция на возвращение медведя
            elif self.auto_step == 2 and self.stage == "bear_spotted_2":
                # УБРАЛИ ОЖИДАНИЕ ЧЕЛОВЕКА! Робот слышит крик и СРАЗУ идет за шашкой
                self.api.command_action(robot.id, "clear")
                self.api.command_move(robot.id, self.flare_pos[0], self.flare_pos[1])
                self.auto_step = 3

            # ЭТАП 3: Дошли до шашки, забираем
            elif self.auto_step == 3 and len(robot.action_queue) == 0:
                mem_flare = next((m for m in robot.memory if m.real_id == self.flare_id), None)
                if mem_flare:
                    self.api.command_action(robot.id, "pickup", mem_flare.id)
                    self.api.command_move(robot.id, self.intercept_pos[0], self.intercept_pos[1], queue=True)
                    self.auto_step = 4

            # ЭТАП 4: Снова у палатки, ждем медведя
            elif self.auto_step == 4 and len(robot.action_queue) == 0:
                if distance(robot, self.bear) < 10.0:
                    mem_flare_inv = next((m for m in robot.memory_inventory if m.real_id == self.flare_id), None)
                    if mem_flare_inv:
                        self.api.command_action(robot.id, "put", mem_flare_inv.id)
                        self.auto_step = 5


        # =====================================================================
        # ПРОВЕРКА ПОРАЖЕНИЯ
        # =====================================================================
        if self.tent.state.get("broken", False) and self.stage not in ["bear_fleeing", "failed"]:
            self.stage = "failed"
            print("\n❌ СЦЕНАРИЙ ПРОВАЛЕН: Медведь разорвал палатку!")