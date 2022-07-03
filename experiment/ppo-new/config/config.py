# Ref: https://github.com/facebookresearch/hydra/blob/1.2_branch/examples/tutorials/structured_configs/5.2_structured_config_schema_different_config_group/database_lib.py
from dataclasses import dataclass, field
from typing import Any, List

from omegaconf import MISSING, OmegaConf  # Do not confuse with dataclass.MISSING

import hydra
from hydra.core.config_store import ConfigStore

from config.base_config import BaseConfig
from config.nam_config import *
from config.tdg_config import *

defaults = [
    # config group name c will load config named base
    {"c": "base"}
]

@dataclass
class Config:
    # this is unfortunately verbose due to @dataclass limitations
    defaults: List[Any] = field(default_factory=lambda: defaults)
    # Hydra will populate this field based on the defaults list
    c: BaseConfig = MISSING


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="c", name="base", node=BaseConfig)
cs.store(group="c", name="convert", node=ConvertConfig)

# other config groups for training
cs.store(group="c", name="nam", node=NamConfig)
cs.store(group="c", name="nam_ft", node=NamFTConfig)
cs.store(group="c", name="nam_pret", node=NamPretrainConfig)
cs.store(group="c", name="nam_mp", node=NamMPConfig)
cs.store(group="c", name="nam_rm_mp", node=NamRMMPConfig)
cs.store(group="c", name="nam_test", node=NamTestConfig)
cs.store(group="c", name="nam_alltest", node=NamAllTestConfig)
cs.store(group="c", name="nam_convert", node=NamConvertConfig)

cs.store(group="c", name="tdg", node=TdgConfig)
cs.store(group="c", name="tdg_ft", node=TdgFTConfig)
cs.store(group="c", name="tdg_mp", node=TdgMPConfig)
cs.store(group="c", name="tdg_rmmp", node=TdgRMMPConfig)

# cfg groups for test
# cs.store(group="c", name="nam_test", node=NamMultiPretrainConfig)
