from meta_agent.sandbox import DockerSandboxExecutor


class DockerRuntime:

    def __init__(self, executor: DockerSandboxExecutor) -> None:
        self.executor = executor

    def register(
        self,
        component_id: str,
        source_code: str,
        entrypoint: str,
        timeout_sec: float | None = None,
    ) -> None:
        """Регистрирует компонент в sandbox.

        ``timeout_sec`` учитывается только при ``invoke``.
        """
        del timeout_sec  # register использует default request timeout
        self.executor.register(
            component_id=component_id,
            source_code=source_code,
            entrypoint=entrypoint,
        )


    def invoke(
        self,
        component_id: str,
        arguments: dict,
        timeout_sec: float | None = None,
    ):
        return self.executor.invoke(
            component_id=component_id,
            arguments=arguments,
            timeout_sec=timeout_sec,
        )

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
