from typing import Any

import logging
from kedro.io.core import AbstractDataset, DatasetError
from comet_ml.api import API

logger = logging.getLogger(__name__)


class CometMLDataset(AbstractDataset):
    def __init__(
        self,
        credentials: dict[str, Any] = None,
        workspace: str = None,
        name: str = None,
        model_version: str = None,
        path: str = None,
    ):
        super().__init__()
        if not (credentials and "token" in credentials and credentials["token"]):
            raise DatasetError(
                "'token' argument cannot be empty. Please " "provide a Comet ML token."
            )
        self.workspace = workspace
        self.name = name
        self.model_version = model_version
        self.path = path

        self.api = API(api_key=credentials["token"])

    def _describe(self):
        return {
            "name": self.name,
            "workspace": self.workspace,
            "model_version": self.model_version,
            "path": self.path,
        }

    def _load(self):
        model = self.api.get_model(self.workspace, self.name)
        model.download(self.model_version, output_folder=self.path)
        return self._describe()

    def _save(self, **kwargs):
        raise DatasetError(f"{self.__class__.__name__} is a read only data set type")

    def _exists(self):
        return self._window()
