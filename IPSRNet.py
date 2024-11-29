# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 20:36:42 2024

@author: Luís Eduardo Sales do Nascimento
"""

from tensorflow.keras.applications import InceptionV3, VGG19, VGG16
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionV3_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from tensorflow.keras import Model 
from tensorflow.keras.models import load_model
import numpy as np
from Utils import load_image, TSS, macroRecall

class IPSR20N():
    def __init__(self):
        self.deepmodel = InceptionV3(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
        self.deepmodel = Model(inputs=self.deepmodel.input, outputs=self.deepmodel.layers[-2].output)
        
        self.finalmodel = load_model('Networks/20m.keras', custom_objects={"TSS":TSS, "macroRecall":macroRecall})
        
    def predict(self, image_path):
        image = load_image(image_path, target_size=(299,299))
        processedimage = inceptionV3_preprocess(image)
        featureMap = np.array(self.deepmodel.predict(processedimage))
        output = self.finalmodel.predict(featureMap)
        return output

class IPSR30N():
    def __init__(self):
        self.deepmodel = VGG19(include_top=True, weights='imagenet', pooling=None, input_shape=(224, 224, 3))
        self.deepmodel = Model(inputs=self.deepmodel.input, outputs=self.deepmodel.layers[-2].output)
        
        self.finalmodel = load_model('Networks/30m.keras', custom_objects={"TSS":TSS, "macroRecall":macroRecall})
        
    def predict(self, image_path):
        image = np.array([load_image(image_path, target_size=(224,224))])
        processedimage = vgg19_preprocess(image)
        featureMap = np.array(self.deepmodel.predict(processedimage))
        output = self.finalmodel.predict(featureMap)
        return output
    
class IPSR60N():
    def __init__(self):
        self.deepmodel = VGG16(include_top=True, weights='imagenet', pooling=None, input_shape=(224, 224, 3))
        self.deepmodel = Model(inputs=self.deepmodel.input, outputs=self.deepmodel.layers[-2].output)
        
        self.finalmodel = load_model('Networks/1h.keras', custom_objects={"TSS":TSS, "macroRecall":macroRecall})
        
    def predict(self, image_path):
        image = load_image(image_path, target_size=(224,224))
        processedimage = image/255
        featureMap = np.array(self.deepmodel.predict(processedimage))
        output = self.finalmodel.predict(featureMap)
        return output

class IPSR120N():
    def __init__(self):
        self.deepmodel = load_model('painters.keras')
        
        self.finalmodel = load_model('Networks/2h.keras', custom_objects={"TSS":TSS, "macroRecall":macroRecall})
        
    def predict(self, image_path):
        image = load_image(image_path, target_size=(256,256))
        
        processedimage = np.zeros_like(image, dtype=np.float32)
        with np.load('painters_preprocessing_stats.npz') as stats:
            mean = np.transpose(stats['mean'], (1, 2, 0))
            std = np.transpose(stats['std'], (1, 2, 0))
            for i in range(image.shape[0]):
                aux = (image[i] - mean)/std
                processedimage[i] = aux
        
        featureMap = np.array(self.deepmodel.predict(processedimage))
        output = self.finalmodel.predict(featureMap)
        return output