# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 20:46:05 2024

@author: luise
"""

from keras import ops
import os
import cv2

def createFolder(caminho):
    """
    Create a folder

    Parameters
    ----------
    caminho : string
        Name of the folder.

    Returns
    -------
    None.
    """
    try:
        if not os.path.exists(caminho):
            os.makedirs(caminho)
    except OSError:
        print ('Error: Creating directory. ' +  caminho)

def recall(y_true, y_pred):
    """
    Get the recall metric.
    
    Parameters
    ----------
    y_true: The ground truth values, with the same dimensions as 'y_pred'.
    y_pred: The predicted values. Each element must be in the range '[0, 1]'.

    Returns
    -------
    recall_keras : The recall value.
    """
    
    true_positives = ops.sum(ops.round(ops.clip(y_true * y_pred, 0, 1)))
    possible_positives = ops.sum(ops.round(ops.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + 1e-07)
    return recall_keras

def load_image(path, target_size=(224,224)):
    """
    Parameters
    ----------
    path : string
        Path to the saved image file.
    target_size : tuple, optional
        New size for resize the image. The default is (224,224).

    Returns
    -------
    img : array
        Output image.
    """
    
    img = cv2.imread(path)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img

def TSS(y_true, y_pred):
    """
    Get the adapted True Skill Statistic (TSS) metric.
    
    Parameters
    ----------
    y_true: The ground truth values, with the same dimensions as 'y_pred'.
    y_pred: The predicted values. Each element must be in the range '[0, 1]'.

    Returns
    -------
    TSS : The adapted TSS value.
    """
    
    # recall for negative class
    Neg_true = y_true[:,0]
    Neg_pred = y_pred[:,0]
    Neg_recall = recall(Neg_true,Neg_pred)
    
    # recall for positive FF class
    FF_true = y_true[:,1]
    FF_pred = y_pred[:,1]
    FF_recall = recall(FF_true,FF_pred)
    
    # recall for positive FR class
    FR_true = y_true[:,2]
    FR_pred = y_pred[:,2]
    FR_Recall = recall(FR_true,FR_pred)
    
    detection_rate = (FF_recall+FR_Recall)/2
    TSS = detection_rate + Neg_recall - 1
    
    return TSS

def macroRecall(y_true, y_pred):
    """
    Get the macro recall metric.
    
    Parameters
    ----------
    y_true: The ground truth values, with the same dimensions as 'y_pred'.
    y_pred: The predicted values. Each element must be in the range '[0, 1]'.

    Returns
    -------
    macroRecall : The macro recall value.
    """
    
    # recall for negative class
    Neg_true = y_true[:,0]
    Neg_pred = y_pred[:,0]
    Neg_recall = recall(Neg_true,Neg_pred)
    
    # recall for positive FF class
    FF_true = y_true[:,1]
    FF_pred = y_pred[:,1]
    FF_recall = recall(FF_true,FF_pred)
    
    # recall for positive FR class
    FR_true = y_true[:,2]
    FR_pred = y_pred[:,2]
    FR_Recall = recall(FR_true,FR_pred)
    
    macroRecall = (Neg_recall+FF_recall+FR_Recall)/3
    
    return macroRecall