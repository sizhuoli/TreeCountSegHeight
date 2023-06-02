#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 00:22:48 2021

@author: sizhuo
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # first gpu

from core2.training_segcount import trainer

# whether using input of the same resolution or multiple resolution (only allow half reduced resolution for now)
multi_input_resolution = False
if multi_input_resolution:
     from config import UNetTraining_multires as configs
else:
     from config import UNetTraining as configs



config = configs.Configuration()
trainer_segcount = trainer(config)
trainer_segcount.vis()
trainer_segcount.train_config()
trainer_segcount.LOAD_model()
trainer_segcount.train()
