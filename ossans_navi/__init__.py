import logging
import logging.config
import os
import tomllib
from typing import Any

LOG_LEVEL = logging.getLevelNamesMapping().get(os.environ.get("OSN_LOG_LEVEL", ""), logging.INFO)

# pyproject.toml から読み込んで logging の設定
with open('pyproject.toml', mode='rb') as fp:
    pyproject: dict[str, Any] = tomllib.load(fp)
    logging_config: dict[str, Any] = pyproject.get('tool', {}).get('logging', {})
    logging.config.dictConfig(logging_config)
    logging.root.setLevel(LOG_LEVEL)
