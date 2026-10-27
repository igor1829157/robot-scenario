from pydantic import BaseModel, Field


class SceneObject(BaseModel):
    object: str = Field(description="Имя объекта")
    shape: list[float] = Field(description="Форма объекта в виде списка локальных координат вершин")
    affordance: list[str] = Field(description="Что может делать робот с этим объектом")
    behavior: list[str] = Field(description="Что может делать объект")
    description: str = Field(description="Описание объекта")
    dynamic: bool = Field(description="Объект динамический или статический")
    has_inventory: bool = Field(description="У объекта есть инвентарь")
    inventory: list[str] = Field(description="Объекты, которые лежат внутри объекта")
    pos: list[float] = Field(description="Положение объекта в виде координат X и Y, если объект не находится в инвентаре у другого объекта")


class ListObject(BaseModel):
    objects: list[SceneObject] = Field(description="Список всех объектов расширенной сцены")
