# load configs from yaml
# load functions
# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from core2.model_manager import ModelManager

import hydra


@hydra.main(config_path='config', config_name='hyperps')
def main(cfg):

    model_manager = ModelManager()

    model_manager.load_files(cfg)
    model_manager.download_weights(cfg)
    model_manager.load_model(cfg)
    model_manager.predict(cfg)








if __name__ == "__main__":
    main()
