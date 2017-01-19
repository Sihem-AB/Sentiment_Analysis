import sklearn
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle


def precision(Y, Ypred):
    return np.sum((Y == Ypred).astype(int)) / float(len(Y))

X_train_doc2vec = pickle.load(open("../X_train_doc2vec.pickle", "rb"))
Y_train = pickle.load(open("../Y_train.pickle", "rb"))

X_test_doc2vec = pickle.load(open("../X_test_doc2vec.pickle", "rb"))
Y_test = pickle.load(open("../Y_test.pickle", "rb"))


clf = LogisticRegression()

clf.fit(X_train_doc2vec, Y_train)

predict = clf.predict(X_test_doc2vec)


score = precision(Y_test, predict)

print ("score : ", score)