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
        """
        Initializes a IPSRNet instance, for 20 minutes of observed data.
        
        This instance consists of 2 neural networks, the first one is the 
        Inception-V3 used to extract features from the images of solar wind 
        and interplanetary magnetic field parameters and the second one is an 
        MLP trained to classify whether there is an interplanetary shock wave 
        FF, FR or none.
        """
        self.deepmodel = InceptionV3(include_top=True, weights='imagenet', input_shape=(299, 299, 3))
        self.deepmodel = Model(inputs=self.deepmodel.input, outputs=self.deepmodel.layers[-2].output)
        
        self.finalmodel = load_model('Networks/20m.keras', custom_objects={"TSS":TSS, "macroRecall":macroRecall})
        
    def predict(self, image_path):
        """
        Parameters
        ----------
        image_path : string
            Path to the saved image file. 

        Returns
        -------
        output : array
            Output array with probability of each class for the sample. 
            The array follows the format [Prob Neg, Prob FF, Prob FR]
        """
        image = np.array([load_image(image_path, target_size=(299,299))])
        processedimage = inceptionV3_preprocess(image)
        featureMap = np.array(self.deepmodel.predict(processedimage, verbose=0))
        output = self.finalmodel.predict(featureMap, verbose=0)[0]
        return output

class IPSR30N():
    def __init__(self):
        """
        Initializes a IPSRNet instance, for 30 minutes of observed data.
        
        This instance consists of 2 neural networks, the first one is the 
        VGG19 used to extract features from the images of solar wind 
        and interplanetary magnetic field parameters and the second one is an 
        MLP trained to classify whether there is an interplanetary shock wave 
        FF, FR or none.
        """
        self.deepmodel = VGG19(include_top=True, weights='imagenet', pooling=None, input_shape=(224, 224, 3))
        self.deepmodel = Model(inputs=self.deepmodel.input, outputs=self.deepmodel.layers[-2].output)
        
        self.finalmodel = load_model('Networks/30m.keras', custom_objects={"TSS":TSS, "macroRecall":macroRecall})
        
    def predict(self, image_path):
        """
        Parameters
        ----------
        image_path : string
            Path to the saved image file. 

        Returns
        -------
        output : array
            Output array with probability of each class for the sample. 
            The array follows the format [Prob Neg, Prob FF, Prob FR]
        """
        image = np.array([load_image(image_path, target_size=(224,224))])
        processedimage = vgg19_preprocess(image)
        featureMap = np.array(self.deepmodel.predict(processedimage, verbose=0))
        output = self.finalmodel.predict(featureMap, verbose=0)[0]
        return output
    
class IPSR60N():
    def __init__(self):
        """
        Initializes a IPSRNet instance, for 60 minutes of observed data.
        
        This instance consists of 2 neural networks, the first one is the 
        VGG16 used to extract features from the images of solar wind 
        and interplanetary magnetic field parameters and the second one is an 
        MLP trained to classify whether there is an interplanetary shock wave 
        FF, FR or none.
        """
        self.deepmodel = VGG16(include_top=True, weights='imagenet', pooling=None, input_shape=(224, 224, 3))
        self.deepmodel = Model(inputs=self.deepmodel.input, outputs=self.deepmodel.layers[-2].output)
        
        self.finalmodel = load_model('Networks/1h.keras', custom_objects={"TSS":TSS, "macroRecall":macroRecall})
        
    def predict(self, image_path):
        """
        Parameters
        ----------
        image_path : string
            Path to the saved image file. 

        Returns
        -------
        output : array
            Output array with probability of each class for the sample. 
            The array follows the format [Prob Neg, Prob FF, Prob FR]
        """
        image = np.array([load_image(image_path, target_size=(224,224))])
        processedimage = image/255
        featureMap = np.array(self.deepmodel.predict(processedimage, verbose=0))
        output = self.finalmodel.predict(featureMap, verbose=0)[0]
        return output

class IPSR120N():
    def __init__(self):
        """
        Initializes a IPSRNet instance, for 30 minutes of observed data.
        
        This instance consists of 2 neural networks, the first one is the 
        Painters used to extract features from the images of solar wind 
        and interplanetary magnetic field parameters and the second one is an 
        MLP trained to classify whether there is an interplanetary shock wave 
        FF, FR or none.
        """
        self.deepmodel = load_model('Networks/painters.keras')
        
        self.finalmodel = load_model('Networks/2h.keras', custom_objects={"TSS":TSS, "macroRecall":macroRecall})
        
    def predict(self, image_path):
        """
        Parameters
        ----------
        image_path : string
            Path to the saved image file. 

        Returns
        -------
        output : array
            Output array with probability of each class for the sample. 
            The array follows the format [Prob Neg, Prob FF, Prob FR]
        """
        image = np.array([load_image(image_path, target_size=(256,256))])
        
        processedimage = np.zeros_like(image, dtype=np.float32)
        with np.load('painters_preprocessing_stats.npz') as stats:
            mean = np.transpose(stats['mean'], (1, 2, 0))
            std = np.transpose(stats['std'], (1, 2, 0))
            for i in range(image.shape[0]):
                aux = (image[i] - mean)/std
                processedimage[i] = aux
        
        featureMap = np.array(self.deepmodel.predict(processedimage, verbose=0))
        output = self.finalmodel.predict(featureMap, verbose=0)[0]
        return output