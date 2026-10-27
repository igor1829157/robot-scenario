import ast


SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


FORBIDDEN_NODE_TYPES = {
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Nonlocal,
    ast.Try,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Yield,
    ast.YieldFrom,
}


FORBIDDEN_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "memoryview",
    "open",
    "setattr",
    "type",
    "vars"
}


def validate_source(
    source_code: str,
    entrypoint: str,
):

    if not entrypoint.isidentifier():
        raise ValueError("entrypoint must be a valid identifier")

    if entrypoint.startswith("_"):
        raise ValueError("entrypoint cannot start with an underscore")

    tree = ast.parse(source_code, "exec")

    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    ]

    other_nodes = [
        node
        for node in tree.body
        if not isinstance(node, ast.FunctionDef)
        and not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
    ]

    if other_nodes:
        raise ValueError("only one top-level function and an optional docstring are allowed")

    if len(functions) != 1:
        raise ValueError("source must contain exactly one top-level function")

    if functions[0].name != entrypoint:
        raise ValueError(f"expected entrypoint {entrypoint!r}, found {functions[0].name!r}")

    for node in ast.walk(tree):
        if type(node) in FORBIDDEN_NODE_TYPES:
            raise ValueError(f"forbidden syntax: {type(node).__name__}")
        
        if (
            isinstance(node, ast.FunctionDef)
            and node is not functions[0]
        ):
            raise ValueError("nested functions are forbidden")

        if isinstance(node, ast.Name):
            if (
                node.id in FORBIDDEN_NAMES
                or node.id.startswith("__")
            ):
                raise ValueError(f"forbidden name: {node.id}")

        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise ValueError(f"forbidden attribute: {node.attr}")
            
            if node.attr in ("format", "format_map"):
                raise ValueError(f"forbidden attribute: {node.attr}")
