#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 20:07:16 2021

@author: sizhuo
"""

# segcount density


import os

import ipdb

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # first gpu
from core2.finetune_segcount import trainer
from config import UNetTrainingFinetune


config = UNetTrainingFinetune.Configuration()
finetuner = trainer(config)
finetuner.load_local_data()
finetuner.load_pretraining_data()
finetuner.wrap_data()
finetuner.model_ready_train()
