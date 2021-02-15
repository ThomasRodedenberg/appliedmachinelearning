#NCC 

import numpy as np

class myNCC :

    def __init__(self):
        self.clusters = None
        self.cluster_labels = []

    def fit(self, data, labels):
        classes = list(set(labels))

        self.clusters = np.zeros([len(classes), len(data[0])])
        for i, j in enumerate(classes):
            self.clusters[i] = (self.cluster(data, labels,j))
            self.cluster_labels.append(j)

    def cluster(self,data, labels, label):
        data_with_label = []
        for i in range (len(data)):
            if labels[i] == label:
                data_with_label.append(data[i])
        return np.mean(data_with_label, axis =0)

    def predict(self, data):
        predictions = np.zeros(len(data))
        for i,x in enumerate(data):
            predictions[i] = (np.argmin(np.linalg.norm(self.clusters - x, axis=1)))
        return predictions
        
