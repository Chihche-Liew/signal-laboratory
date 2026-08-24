"""Configuration loading for siglab.

The default config ships INSIDE the package (siglab/configs/default.yaml) and
is loaded via importlib.resources, so it works from a checkout, an editable
install, or a wheel — independent of the current working directory.
"""

from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

_config: dict[str, Any] | None = None


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration, caching the result.

    With ``path`` given, that file is loaded (and cached). Without it, the
    packaged default is used.
    """
    global _config
    if _config is not None and path is None:
        return _config
    if path is not None:
        text = Path(path).read_text()
    else:
        text = (files("siglab") / "configs" / "default.yaml").read_text()
    _config = yaml.safe_load(text)
    return _config
