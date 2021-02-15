#NBC 

import numpy as np

class myNBC :

    def __init__(self):
        self.information = {}
        self.classes = None
        self.p_per_class = None


    def fit(self, data, labels):
        self.classes, cnt = np.unique(labels, return_counts=True)
        self.cnt_per_class = dict(zip(self.classes, cnt))
        self.p_per_class = dict(zip(self.classes, cnt/len(labels)))
        
        for i, j in enumerate(self.classes):
            self.information[j] = (self.organize_information(data, labels,j))


    def organize_information(self,data, labels, label):
        data_with_label = []
        for i in range (len(data)):
            if labels[i] == label:
                data_with_label.append(data[i])
                
        data_with_label = np.asarray(data_with_label)
        #variance_for_label = np.var(data_with_label, axis =0)
        information = {}
        for i in range(len(data[0])):
            values, nbr = np.unique(data_with_label[:,i],return_counts=True, axis = 0)
            information[i] = dict(zip(values, nbr/self.cnt_per_class[label]))
        return information

    def predict(self, data):
        predictions = np.zeros(len(data))
        for i,x in enumerate(data):
            probability_per_class = np.zeros(len(self.classes))
            for c in self.classes:
                probability_per_class[c] = self.p_per_class[c]
                for p,v in enumerate(x):
                    if v in self.information[c][p]:
                        probability_per_class[c] *= self.information[c][p][v]
                    else:
                        probability_per_class[c] *= 0
            predictions[i] = np.argmax(probability_per_class)
        return predictions
        
