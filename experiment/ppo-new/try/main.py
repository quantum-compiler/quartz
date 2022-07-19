import hydra
from config.config import *


@hydra.main(config_path='config', config_name="config")
def my_app(cfg: Config) -> None:
    # print(OmegaConf.to_yaml(cfg.group))
    print(dict(cfg))


if __name__ == "__main__":
    my_app()
