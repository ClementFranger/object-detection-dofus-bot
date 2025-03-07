from typing import Any

from supervision import Detections

Obs = dict[str, Detections | bool | Any] | None
