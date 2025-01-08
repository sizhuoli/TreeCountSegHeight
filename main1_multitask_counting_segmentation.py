#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 00:22:48 2021
@author: sizhuo
"""

import os
from core2.training_segcount import Trainer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# whether using input of the same resolution or multiple resolution (only allow half reduced resolution for now)
multi_input_resolution = False

if multi_input_resolution:
     from config import UNetTraining_multires as configs
else:
     from config import UNetTraining as configs

config = configs.Configuration()
trainer_segcount = Trainer(config)
trainer_segcount.visualize_patches()
trainer_segcount.configure_training()
trainer_segcount.load_model()
trainer_segcount.train()
