import os

from rubix.utils import read_yaml

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
RUBIX_CONFIG_PATH = os.path.join(PARENT_DIR, "rubix_config.yml")
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "rubix_config.yml"
)

PIPELINE_CONFIG_PATH = os.path.join(PARENT_DIR, "pipeline_config.yml")


class Config:

    @staticmethod
    def load() -> dict:
        rubix_config = read_yaml(RUBIX_CONFIG_PATH)
        pipeline_config = read_yaml(PIPELINE_CONFIG_PATH)
        config = {**rubix_config, "pipelines": pipeline_config}
        return config
