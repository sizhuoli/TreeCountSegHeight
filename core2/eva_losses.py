#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:55:03 2021

@author: sizhuo
"""

from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf

def eva_acc(y_true, y_label):
    
    return accuracy_score(y_true.flatten(), y_label.flatten())

def eva_dice(y_true, y_pred):
    # same as F1
    intersection = np.sum(np.abs(y_true.flatten() * y_pred.flatten()))
    if (np.sum(y_true)==0) and (np.sum(y_pred)==0):
        return 1
    return (2*intersection) / (np.sum(y_true) + np.sum(y_pred))



def true_positives(y_true, y_pred):
    """compute true positive"""
    return np.round(y_true * y_pred)

def false_positives(y_true, y_pred):
    """compute false positive"""
    
    return np.round((1 - y_true) * y_pred)

def true_negatives(y_true, y_pred):
    """compute true negative"""
    return np.round((1 - y_true) * (1 - y_pred))

def false_negatives(y_true, y_pred):
    """compute false negative"""

    return np.round((y_true) * (1 - y_pred))

def eva_sensitivity(y_true, y_pred):
    """compute sensitivity (recall)"""
    tp = true_positives(y_true, y_pred)
    fn = false_negatives(y_true, y_pred)
    return np.sum(tp) / (np.sum(tp) + np.sum(fn))

def eva_specificity(y_true, y_pred):
    """compute specificity (precision)"""
    tn = true_negatives(y_true, y_pred)
    fp = false_positives(y_true, y_pred)
    return np.sum(tn) / (np.sum(tn) + np.sum(fp))

def eva_miou(y_true, y_pred):
    mioufuc = tf.keras.metrics.MeanIoU(num_classes=2)
    mioufuc.update_state(y_true, y_pred)
    return mioufuc.result().numpy()

