from sklearn import metrics, datasets
from sklearn.naive_bayes import GaussianNB
import numpy as np
import myNCC
import myNBC
import myGNB
import myEM


def main() :


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
    classes = [0,1,2,3,4,5,6,7,8,9]



    clusters = np.zeros(len(classes))
    em = myEM.myEM(eps = 0.05)
    em.EM_GMM(test_features, classes)
    #em.fit(train_features, train_labels)
    p = em.predict(test_features)



    print("Classification report SKLearn GNB:\n%s\n"
      % (metrics.classification_report(test_labels, p)))
    print("Confusion matrix SKLearn GNB:\n%s" % metrics.confusion_matrix(test_labels, p))


if __name__ == "__main__": main()
