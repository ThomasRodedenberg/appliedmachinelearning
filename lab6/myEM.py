#EM

import numpy as np

class myEM :

    def __init__(self, eps = 0.01, seed = 20):
        self.information = {}
        self.classes = None
        self.eps = eps
        self.mu = {}
        self.cov = {}
        self.class_prior = None
        self.seed = seed

    def EM_GMM(self, data, classes):
        self.classes = classes
        self.class_prior = np.ones(len(classes))/len(classes)
        np.random.seed(self.seed)

        idxs = np.resize(range(len(classes)), len(data))
        np.random.shuffle(idxs)
        for j in self.classes:
            self.mu[j] = np.mean(data[idxs == j], axis=0)
            self.cov[j] = np.var(data[idxs == j], axis=0)+ self.eps

        boole = True
        while boole:
            boole = self.em(data)
        
    def em(self,data):
        #E
        r= {}
        for i,image in enumerate(data):
            r_c = {}
            r_n = {}
            #r = np.zeros(len(self.classes))
            for j,c in enumerate(self.classes):
                r_n[c] = self.class_prior[j]
                for pixel,value in enumerate(image):
                   r_n[c] *= self.GNB_probability(pixel,c,value)
            for c in self.classes:
                r_c[c]= r_n[c]/ sum(r_n.values())
            r[i] = r_c
        return self.M_(data,r)

    def M_(self,data, r):
        new_class_prior  = self.class_prior
        r_k = {key: 0 for key in self.classes}
        for j,c in enumerate(self.classes):
            mu = np.zeros(len(data[0]))
            cov = np.zeros(len(data[0]))
            for i in range(len(r)):
                r_k[c] += r[i][c]
                mu += r[i][c] * data[i]
                cov += r[i][c] * data[i] * data[i].transpose()
            self.mu[c] = mu/r_k[c]
            self.cov[c] = cov/r_k[c] - self.mu[c]*self.mu[c].transpose() + self.eps
            new_class_prior[j] = r_k[c]/len(data)
        if np.linalg.norm(self.class_prior - new_class_prior) < 1e-4:
            self.class_prior = new_class_prior
            return False
        self.class_prior = new_class_prior
        return True
            

    def GNB_probability(self, pixel, c, value):
        numerator = np.exp(-((value-self.mu[c][pixel])**2)/(2*self.cov[c][pixel]**2))
        denominator = np.sqrt(2*np.pi*self.cov[c][pixel]**2)
        #numerator = np.exp(-((value-self.mu[c][pixel])**2)/(2*self.cov[c][pixel]))
        #denominator = np.sqrt(2*np.pi*self.cov[c][pixel])
        return numerator/denominator

    def predict(self, data):
        predictions = np.zeros(len(data))
        for i,x in enumerate(data):
            probability_per_class = np.zeros(len(self.classes))
            for c in self.classes:
                probability_per_class[c] = self.class_prior[c]
                for p,v in enumerate(x):
                   probability_per_class[c] *= self.GNB_probability(p,c,v)
            predictions[i] = np.argmax(probability_per_class)
        print(probability_per_class)
        return predictions
        
