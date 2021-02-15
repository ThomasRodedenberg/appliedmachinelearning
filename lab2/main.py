import ToyData as td
import ID3
from collections import Counter, OrderedDict

import numpy as np
from sklearn import tree, metrics, datasets

from sklearn import tree, metrics, datasets
import matplotlib.pyplot as plt



def main():
    digits = datasets.load_digits()
    num_examples = len(digits.data)
    num_split = int(0.7*num_examples)

    if True:
        for n in digits.data:
            for i in range( len(n)):
                if n[i] <= 5:
                    n[i] = 5
                elif n[i] > 5 and i <= 11:
                    n[i] = 11
                else:
                    n[i] = 16

    train_features = digits.data[:num_split]
    train_labels =  digits.target[:num_split]
    test_features = digits.data[num_split:]
    test_labels = digits.target[num_split:]
    #classes2 = ('0', '1', '2','3','4','5','6','7','8','9')
    classes2 = (0, 1, 2,3,4,5,6,7,8,9)

    print(test_features[0])

    attributes2 = OrderedDict()
    for i in range(64):
        attributes2[i] = [float(j) for j in range(16)]
        
    attributes3 = OrderedDict()
    for i in range(64):
        attributes3[i] = [5,11,16]
    
    

    #print(attributes)
    #print(attributes3)
    print(len(train_features))   


    attributes, classes, data, target, data2, target2 = td.ToyData().get_data()

    id3 = ID3.ID3DecisionTreeClassifier()

    myTree = id3.fit(train_features, train_labels, attributes3, classes2)
    
    #myTree = id3.fit(train_features, train_labels, attributes2, classes2)


    #myTree = id3.fit(data, target, attributes, classes)
    #print(myTree)
    #plot = id3.make_dot_data()
    #plot.render("testTree")
    #predicted = id3.predict(data2, myTree)
    predicted = id3.predict(test_features, myTree)
    print('predicted')
    print(predicted)    


    #print("Confusion matrix:\n%s" % metrics.confusion_matrix(target2, predicted))
    
    print(metrics.confusion_matrix(test_labels, predicted))


if __name__ == "__main__": main()
