#!/usr/bin/python3

import Utils
from Preprocessing import do_preprocessing

import numpy as np
try:
   import _pickle as pickle
except:
   import pickle

def save_aspickles(variables, filenames, rep_pickle):
	for i in range(len(variables)):
		f = open(rep_pickle + filenames[i], "wb")
		pickle.dump(variables[i], f)
		f.close()


def load_pickles(filenames, rep_pickle):
	variables = []
	for i in range(len(filenames)):
		variables.append(pickle.load(open(rep_pickle + filenames[i], "rb")))
	return variables

"""
 # DB_ONE: Sentence Polarity Dataset
	This dataset is made up of 2 files: rt-polarity.neg and rt-polarity.pos

 #DB_TWO: Large Movie Review Dataset
	This dataset is made up of 2 directories: pos/ and neg/. And each directory contains a number of review files
"""

# --------------------------Parameters --------------------------------------------------------------------------------
pos_path = "D:/Users/abdel/rt-polaritydata/rt-polarity.neg"
neg_path = "D:/Users/abdel/rt-polaritydata/rt-polarity.pos"

selected_DB = Utils.DB_ONE

is_bigrams = False
method = "MI"
vector_type = "TF-IDF"
vector_type = "TF-IDF-SENTIWORDNET"
# vector_type = "FREQ"

#############################################################################################
# 1st use case: when necessary json files are not created yet
#############################################################################################

rep_pickle = "../dataset1_pickle/"

# --------------------------------------------- MODELS-------------------------------------------------
k = 0.1 # % pourcentage of word we keep by mutual information
# vocabs, reduced_vocabs, fs, features_space = do_preprocessing(pos_path, neg_path, selected_DB, is_bigrams, k, method)

filenames_pickle = ["vocabs_train"+str(k)+".pickle", "reduced_vocabs_train"+str(k)+".pickle", "fs_train"+str(k)+".pickle", "featurespace"+str(k)+".pickle"]
vocabs, reduced_vocabs, fs, features_space = load_pickles(filenames_pickle, rep_pickle)

# save_aspickles([vocabs, reduced_vocabs, fs, features_space], filenames=filenames_pickle, rep_pickle=rep_pickle)


print("debut tfidf train")
X_tfidf, Y = fs.create_bag_of_words_model(reduced_vocabs, features_space, vector_type="TF-IDF")

print ("debut tfidf sentiwordnet")
X_tfidf_sentiwordnet, Y = fs.create_bag_of_words_model(reduced_vocabs, features_space, vector_type="TF-IDF-SENTIWORDNET")

print("debut doc2vec train")
model_doc2tovec, X_doc2vec, Y = fs.create_doc2vec_model(vocabs, size=400)

print("debut doc2vectfidf train")
model_doc2tovec, X_doc2vec_tfidf, Y = fs.create_doc2vec_tfidf_model(vocabs, reduced_vocabs, features_space)

X_tfidf_sentiwordnet, Y = fs.create_bag_of_words_model(reduced_vocabs, features_space, vector_type="TF-IDF-SENTIWORDNET")

# We split into two differents dataset
indices = np.random.permutation(X_tfidf.shape[0])
size_train = int(0.8 *X_tfidf.shape[0])
training_idx, test_idx = indices[:size_train], indices[size_train:]
X_train_doc2vec, X_test_doc2vec = X_doc2vec[training_idx,:], X_doc2vec[test_idx,:]
X_train_doc2vec_tfidf, X_test_doc2vec_tfidf = X_doc2vec_tfidf[training_idx,:], X_doc2vec_tfidf[test_idx,:]
X_train_tfidf, X_test_tfidf = X_tfidf[training_idx,:], X_tfidf[test_idx,:]
X_train_tfidf_sentiwordnet, X_test_tfidf_sentiwordnet = X_tfidf_sentiwordnet[training_idx,:], X_tfidf_sentiwordnet[test_idx,:]
Y_train, Y_test =  Y[training_idx], Y[test_idx]

# We save as a pickle
filenames_pickle = ["X_train_doc2vec.pickle", "X_train_doc2vec_tfidf"+str(k)+".pickle", "X_train_tfidf" + str(k)+".pickle", "X_train_tfidf_sentiwordnet" + str(k)+".pickle", "Y_train.pickle"]
save_aspickles([X_train_doc2vec, X_train_doc2vec_tfidf, X_train_tfidf,X_train_tfidf_sentiwordnet, Y_train], filenames=filenames_pickle, rep_pickle=rep_pickle)


filenames_pickle = ["X_test_doc2vec.pickle", "X_test_doc2vec_tfidf"+str(k)+".pickle", "X_test_tfidf" + str(k)+".pickle", "X_test_tfidf_sentiwordnet"+str(k)+".pickle", "Y_test.pickle"]
save_aspickles([X_test_doc2vec, X_test_doc2vec_tfidf, X_test_tfidf, X_test_tfidf_sentiwordnet, Y_test], filenames=filenames_pickle, rep_pickle=rep_pickle)