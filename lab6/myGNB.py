#GNB

import numpy as np
from scipy.stats import norm

class myGNB :

    def __init__(self, eps = 0.0001):
        self.mean = {}
        self.variance = {}
        self.classes = None
        self.p_per_class = None
        self.eps = eps


    def fit(self, data, labels):
        self.classes, cnt = np.unique(labels, return_counts=True)
        self.cnt_per_class = dict(zip(self.classes, cnt))
        self.p_per_class = dict(zip(self.classes, cnt/len(labels)))
        
        for i, j in enumerate(self.classes):
            mean,var = (self.organize_information(data, labels,j))
            self.mean[j] = mean
            self.variance[j] = var


    def organize_information(self,data, labels, label):
        data_with_label = []
        for i in range (len(data)):
            if labels[i] == label:
                data_with_label.append(data[i])
                
        mean_for_label = np.mean(data_with_label, axis =0)
        variance_for_label = np.std(data_with_label, axis =0) + self.eps #?
        
        return mean_for_label, variance_for_label


    def predict(self, data):
        predictions = np.zeros(len(data))
        for i,x in enumerate(data):
            probability_per_class = np.zeros(len(self.classes))
            for c in self.classes:
                probability_per_class[c] = self.p_per_class[c]
                for p,v in enumerate(x):
                   probability_per_class[c] *= self.GNB_probability(p,c,v)
            predictions[i] = np.argmax(probability_per_class)
        print(probability_per_class)
        return predictions

    def GNB_probability(self, pixel, c, value):
        numerator = np.exp(-((value-self.mean[c][pixel])**2)/(2*self.variance[c][pixel]**2))
        denominator = np.sqrt(2*np.pi*self.variance[c][pixel]**2)
        #numerator = np.exp(-((value-self.mean[c][pixel])**2)/(2*self.variance[c][pixel]))
        #denominator = np.sqrt(2*np.pi*self.variance[c][pixel])
        return numerator/denominator

