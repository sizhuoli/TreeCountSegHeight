#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 00:22:48 2021

@author: sizhuo
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

from core2.training_segcount import trainer

from config import UNetTraining
config = UNetTraining.Configuration()

trainer_segcount = trainer(config)
trainer_segcount.vis()
trainer_segcount.train_config()
if 'complex' in config.model_name:
    print('complex model')
    trainer_segcount.LOAD_model()

trainer_segcount.train()
trainer_segcount.train_retrain()
trainer_segcount.train_retrain_eff()





