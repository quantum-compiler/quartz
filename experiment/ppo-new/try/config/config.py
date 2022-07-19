# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from ast import Sub
from dataclasses import dataclass, field
from typing import Any, List
from omegaconf import MISSING, OmegaConf  # Do not confuse with dataclass.MISSING

import hydra
from hydra.core.config_store import ConfigStore
import pydantic

@dataclass
class Sub():
    suba: str = 'suba_config'

@dataclass
class Base():
    
    a: int = 1
    b: str = '2'
    sub: Sub = Sub('sub_base')


@dataclass
class Child(Base):
    c: float = 3.14
    b: str = 'child_b'
    sub: Sub = Sub('sub_child')
    l: List[Any] = field(default_factory=lambda: ['list1', 'list2'])


defaults = [
    # config group name db will load config named mysql
    {"group": "base"}
]

@dataclass
class Config():
    # this is unfortunately verbose due to @dataclass limitations
    defaults: List[Any] = field(default_factory=lambda: defaults)

    # Hydra will populate this field based on the defaults list
    group: Base = MISSING



cs = ConfigStore.instance()
cs.store(group="group", name="base", node=Base)
cs.store(group="group", name="child", node=Child)
cs.store(name="config", node=Config)

