#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for segmentation evaluation and tree density analysis.

Created on Mon Jun 19 22:41:28 2023
@author: sizhuo
"""

import os
import numpy as np
from sklearn.metrics import mean_absolute_error
from config_cisong import Model_compare_multires
from core2.model_compare import Eva_segcount

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

config = Model_compare_multires.Configuration()

evaluator = Eva_segcount(config)

evaluator.predict(threshold=0.5)
(
    all_counts, no_small_counts, gt_counts, 
    gt_per_ha, no_small_per_ha, clear_predictions, ground_truths
) = evaluator.report_segmentation(threshold=2)

gt_densities, pred_densities, image_areas = evaluator.report_count_density()

mae = mean_absolute_error(gt_densities, pred_densities)
rmae = mae / np.mean(gt_densities)

absolute_total_error = np.abs(np.sum(np.array(gt_densities) - np.array(pred_densities)))
relative_total_error = absolute_total_error / np.sum(gt_densities) * 100

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Relative MAE (RMAE): {rmae:.2%}")
print(f"Relative Total Error (RTE): {relative_total_error:.2f}%")
