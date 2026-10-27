import argparse
import importlib.resources as resources
import os
import pathlib
from typing import Any, Optional

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel

from llm_teller.core.schemes import ListObject
from llm_teller.utils.proxy import detect_proxy


class BaseChainModel:
    def __init__(self, model: str, provider: str, secret_path: str) -> None:
        secrets_path = pathlib.Path(secret_path).expanduser().resolve()

        if not secrets_path.is_file():
            raise FileNotFoundError(
                "Секретный файл не найден. Проверь путь в переменной окружения "
                "`LLM_TELLER_KEY`: "
                f"{secrets_path.name}"
            )
        with open(secrets_path, "r", encoding="utf-8") as file:
            api_key = file.read().strip()

        client_args = {"proxy": detect_proxy()} if detect_proxy() else {}

        self.model = init_chat_model(
            model=model,
            model_provider=provider,
            api_key=api_key,
            client_args=client_args
        )

    def _build_chain(
            self,
            system: str,
            user: str,
            structured_output: Optional[type[BaseModel]] = None,
            temperature: float = 0.7,
        ) -> Any:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system),
                ("user", user),
            ])
            model = self.model.bind(temperature=temperature)
            if structured_output is not None:
                model = model.with_structured_output(structured_output)
                return prompt | model
            else:
                return prompt | model | StrOutputParser()


class Morgan(BaseChainModel):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Цепочка для генерации расширенного описания сцены по короткому описанию
        scene_system = (
            resources.files("llm_teller.prompts") / "system.scene.prompt"
        ).read_text(encoding="utf-8").strip()
        scene_user = "{short_scenario}"

        self.scene_chain = self._build_chain(
            system=scene_system,
            user=scene_user,
            structured_output=None,
            temperature=1.0,
        )
        #

        # Цепочка для генерации описаний списков объектов расширенной сцены
        description_system = (
            resources.files("llm_teller.prompts") / "system.description.prompt"
        ).read_text(encoding="utf-8").strip()
        description_user = "{objects}"
        self.description_chain = self._build_chain(
            system=description_system,
            user=description_user,
            structured_output=ListObject,
            temperature=0.0,
        )
        #


def _infer_provider(model: str) -> str | None:
    """Инференс провайдера по имени модели.
    """
    m = model.lower()
    if "gemini" in m or "google" in m:
        return "google_genai"
    if m.startswith("gpt-") or m.startswith("o1") or m.startswith("o3") or "openai" in m:
        return "openai"
    return None


DEFAULT_TEXT = (
    "Группа ученых расположилась в заснеженной тундре для исследований. "
    "В лесу за деревьями притаился медведь, готовый напасть в любой момент."
)


def _get_secret_path_from_env() -> str:
    """Возвращает путь к файлу секрета из переменной окружения."""
    value = os.environ.get("LLM_TELLER_KEY")
    if value:
        return value
    raise RuntimeError(
        "Не задан путь к файлу секрета. Установите `LLM_TELLER_KEY` "
        "в переменных окружения."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Генерация расширенной сцены (Morgan) из короткого текста.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Имя модели для запроса.",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=False,
        default=DEFAULT_TEXT,
        help="Текст для генерации сцены (вход в chain `scene_chain`).",
    )
    args = parser.parse_args()

    morgan = Morgan(
        model=args.model,
        provider=_infer_provider(args.model),
        secret_path=_get_secret_path_from_env(),
    )

    scene = morgan.scene_chain.invoke({"short_scenario": args.text})
    print(scene)

    objects = morgan.description_chain.invoke({"objects": scene})
    print(objects)
