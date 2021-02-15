from sklearn import metrics, datasets
from sklearn.naive_bayes import GaussianNB
import numpy as np
import MNIST
import myNCC
import myNBC
import myGNB


def main() :
    mnist = MNIST.MNISTData('MNIST_Light/*/*.png')

    digits = datasets.load_digits()
    num_examples = len(digits.data)
    num_split = int(0.7*num_examples)
    
    simple = False
    if simple:
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

    train_features, test_features, train_labels, test_labels = mnist.get_data()

    #print(train_features[0])

    my_gnb = myGNB.myGNB(eps = 0.05)
    my_gnb.fit(train_features, train_labels)
    p = my_gnb.predict(test_features)
    print(len(p))

    gnb = GaussianNB()
    gnb.fit(train_features, train_labels)
    y_pred = gnb.predict(test_features)

    print("Classification report SKLearn GNB:\n%s\n"
      % (metrics.classification_report(test_labels, p)))
    print("Confusion matrix SKLearn GNB:\n%s" % metrics.confusion_matrix(test_labels, p))

    #mnist.visualize_wrong_class(y_pred, 8)

if __name__ == "__main__": main()
