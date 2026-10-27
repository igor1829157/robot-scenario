import os
import urllib


def detect_proxy() -> str | None:
    raw = (
        os.environ.get("ALL_PROXY")
        or os.environ.get("all_proxy")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("http_proxy")
    )
    if not raw:
        return None
    
    if "://" not in raw:
        return f"socks5://{raw}"

    parsed = urllib.parse.urlparse(raw)
    scheme = parsed.scheme

    if scheme.startswith("socks"):
        return f"socks5://{parsed.hostname}:{parsed.port}"
    if scheme in ("http", "https"):
        return raw

    return None