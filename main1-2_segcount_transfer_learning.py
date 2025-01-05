#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segcount density

Created on Mon Jul 12 20:07:16 2021
@author: sizhuo
"""

import os
from core2.finetune_segcount import Trainer
from config import UNetTrainingFinetune

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # first gpu


config = UNetTrainingFinetune.Configuration()
finetuner = Trainer(config)
finetuner.load_local_data()
finetuner.load_pretraining_data()
finetuner.wrap_data()
finetuner.model_ready_train()
