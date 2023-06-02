#    Edited by Sizhuo Li
#    Author: Ankit Kariryaa, University of Bremen


import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf

def tversky(y_true, y_pred, alpha=0.6, beta=0.4):
    """
    Function to calculate the Tversky loss for imbalanced data
    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param alpha: weight of false positives
    :param beta: weight of false negatives
    :param weight_map:
    :return: the loss
    """
    
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    # weights
    y_weights = y_true[...,1]
    y_weights = y_weights[...,np.newaxis]
    
    ones = 1 
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_t
    g1 = ones - y_t

    tp = tf.reduce_sum(y_weights * p0 * g0)
    fp = alpha * tf.reduce_sum(y_weights * p0 * g1)
    fn = beta * tf.reduce_sum(y_weights * p1 * g0)

    EPSILON = 0.00001
    numerator = tp
    denominator = tp + fp + fn + EPSILON
    score = numerator / denominator
    return 1.0 - tf.reduce_mean(score)

def accuracy(y_true, y_pred):
    """compute accuracy"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.equal(K.round(y_t), K.round(y_pred))

def dice_coef(y_true, y_pred, smooth=0.0000001):
    """compute dice coef"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    
    y_t = np.squeeze(K.flatten(y_t))
    y_pred = np.squeeze(K.flatten(y_pred).numpy())
    # print(type(y_t))
    # print('pred',type(y_pred))
    # intersection = K.sum(K.abs(y_t * y_pred), axis=-1)
    # union = K.sum(y_t, axis=-1) + K.sum(y_pred, axis=-1)
    # intersection = K.sum(K.abs(y_t * y_pred))
    intersection = K.sum(K.abs(y_t * y_pred))
    union = K.sum(y_t) + K.sum(y_pred)
    return (2. * intersection + smooth) / (union + smooth)

def dice_loss(y_true, y_pred):
    """compute dice loss"""
    # y_t = y_true[...,0]
    # y_t = y_t[...,np.newaxis]
    return 1 - dice_coef(y_true, y_pred)

def true_positives(y_true, y_pred):
    """compute true positive"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.round(y_t * y_pred)

def false_positives(y_true, y_pred):
    """compute false positive"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.round((1 - y_t) * y_pred)

def true_negatives(y_true, y_pred):
    """compute true negative"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.round((1 - y_t) * (1 - y_pred))

def false_negatives(y_true, y_pred):
    """compute false negative"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    return K.round((y_t) * (1 - y_pred))

def sensitivity(y_true, y_pred):
    """compute sensitivity (recall)"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    tp = true_positives(y_t, y_pred)
    fn = false_negatives(y_t, y_pred)
    return K.sum(tp) / (K.sum(tp) + K.sum(fn))

def specificity(y_true, y_pred):
    """compute specificity (precision)"""
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    tn = true_negatives(y_t, y_pred)
    fp = false_positives(y_t, y_pred)
    return K.sum(tn) / (K.sum(tn) + K.sum(fp))

def miou(y_true, y_pred):
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    mioufuc = tf.keras.metrics.MeanIoU(num_classes=2)
    mioufuc.update_state(y_t, y_pred)
    return mioufuc.result().numpy()


def weight_miou(y_true, y_pred):
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    y_weights = y_true[...,1]
    y_weights = y_weights[...,np.newaxis]
    mioufuc = tf.keras.metrics.MeanIoU(num_classes=2)
    mioufuc.update_state(y_t, y_pred, sample_weight = y_weights)
    return mioufuc.result().numpy()
