import os
import pytest
from kedro.config import OmegaConfigLoader
from pathlib import Path
from unittest.mock import MagicMock

from object_detection_dofus_bot.datasets.cometml import CometMLDataset
from object_detection_dofus_bot.pipelines.botting.env import DofusEnv


@pytest.fixture(scope="session")
def catalog_config():
    """Fixture for the catalog configuration."""
    path = str(Path(__file__).parent.parent / "conf")
    config_loader = OmegaConfigLoader(conf_source=path, env="base")
    return config_loader


@pytest.fixture(scope="session")
def model(catalog_config):
    """Télécharge le modèle depuis Comet ML et retourne l'objet modèle."""
    model_config = catalog_config["catalog"]["model"]
    model_config["credentials"] = {"token": os.environ["COMET_API_KEY"]}

    comet_dataset = CometMLDataset(
        credentials=model_config["credentials"],
        workspace=model_config["workspace"],
        name=model_config["name"],
        model_version=model_config["model_version"],
        path=model_config["path"],
    )
    return comet_dataset._load()


@pytest.fixture(scope="session")
def dofus():
    return MagicMock(left=0, top=0, width=1920, height=1080)


@pytest.fixture(scope="session")
def data():
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def env(model, dofus):
    return DofusEnv(model, dofus)
