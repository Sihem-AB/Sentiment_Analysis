#!/usr/bin/python3
import Utils
from Preprocessing import do_preprocessing

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
rep_train = "D:/Users/abdel/aclImdb/train"
rep_test = "D:/Users/abdel/aclImdb/test"

pos_path_train = rep_train + "/pos/"
neg_path_train = rep_train + "/neg/"

pos_path_test = rep_test + "/pos/"
neg_path_test = rep_test + "/neg/"

# pos_path = "../sampledata/dataset2/pos/"
# neg_path = "../sampledata/dataset2/neg/"

selected_DB = Utils.DB_TWO

is_bigrams = False
method = "MI"
vector_type = "TF-IDF"
# vector_type = "FREQ"
rep_pickle = "../imdb_pickle/"

# --------------------------------------------- MODEL FOR TRAINING_SET -------------------------------------------------
k = 0.1 # % pourcentage of word we keep by mutual information
# vocabs, reduced_vocabs, fs, features_space = do_preprocessing(pos_path_train, neg_path_train, selected_DB, is_bigrams, k, method)

filenames_pickle = ["vocabs_train"+str(k)+".pickle", "reduced_vocabs_train"+str(k)+".pickle", "fs_train"+str(k)+".pickle", "featurespace"+str(k)+".pickle"]
vocabs, reduced_vocabs, fs, features_space = load_pickles(filenames_pickle, rep_pickle)

# save_aspickles([vocabs, reduced_vocabs, fs, features_space], filenames=filenames_pickle, rep_pickle=rep_pickle)

print("debut tfidf train")
X_train_tfidf, Y_train = fs.create_bag_of_words_model(reduced_vocabs, features_space, vector_type="TF-IDF")

print("debut doc2vec train")
model_doc2tovec, X_train_doc2vec, Y_train = fs.create_doc2vec_model(vocabs, size=400)

print("debut doc2vectfidf train")
model_doc2tovec, X_train_doc2vec_tfidf, Y_train = fs.create_doc2vec_tfidf_model(vocabs, reduced_vocabs, features_space)

print("debut tfidf sentiwordnet train")
X_train_tfidf_sentiwordnet, Y_train = fs.create_bag_of_words_model(reduced_vocabs, features_space, vector_type="TF-IDF-SENTIWORDNET")

# We save as a pickle
# filenames_pickle = ["X_train_doc2vec.pickle", "X_train_doc2vec_tfidf"+str(k)+".pickle", "X_train_tfidf" + str(k)+".pickle", "X_train_tfidf_sentiwordnet" + str(k)+".pickle", "Y_train.pickle"]
# save_aspickles([X_train_doc2vec, X_train_doc2vec_tfidf, X_train_tfidf, X_train_tfidf_sentiwordnet, Y_train], filenames=filenames_pickle, rep_pickle=rep_pickle)

filenames_pickle = ["X_train_tfidf_sentiwordnet" + str(k)+".pickle", "Y_train.pickle"]
save_aspickles([X_train_tfidf_sentiwordnet, Y_train], filenames=filenames_pickle, rep_pickle=rep_pickle)

#----------------------------------------------- MODEL FOR TEST SET -------------------------------------------------------------
print ("début preprocessing test")
# vocabs, reduced_vocabs, fs = do_preprocessing(pos_path_test, neg_path_test, selected_DB, is_bigrams, features_space=features_space)

filenames_pickle = ["vocabs_test"+str(k)+".pickle", "reduced_vocabs_test"+str(k)+".pickle", "fs_test"+str(k)+".pickle"]
# save_aspickles([vocabs, reduced_vocabs, fs], filenames=filenames_pickle, rep_pickle=rep_pickle)
vocabs, reduced_vocabs, fs, features_space = load_pickles(filenames_pickle, rep_pickle)


print("debut doc2vec test")
model_doc2tovec, X_test_doc2vec, Y_test = fs.create_doc2vec_model(vocabs, size=400)

print("debut doc2vectfidf test")
model_doc2tovec, X_test_doc2vec_tfidf, Y_test = fs.create_doc2vec_tfidf_model(vocabs,reduced_vocabs, features_space)

print("debut tfidf test")
X_test_tfidf, Y_test = fs.create_bag_of_words_model(reduced_vocabs, features_space, vector_type=vector_type)

print("debut tfidf sentiwordnet test")
X_test_tfidf_sentiwordnet, Y_test = fs.create_bag_of_words_model(reduced_vocabs, features_space, vector_type="TF-IDF-SENTIWORDNET")

# We save as a pickle
filenames_pickle = ["X_test_doc2vec.pickle", "X_test_doc2vec_tfidf"+str(k)+".pickle", "X_test_tfidf" + str(k)+".pickle", "X_test_tfidf_sentiwordnet"+str(k)+".pickle", "Y_test.pickle"]
save_aspickles([X_test_doc2vec, X_test_doc2vec_tfidf, X_test_tfidf, X_test_tfidf_sentiwordnet, Y_test], filenames=filenames_pickle, rep_pickle=rep_pickle)


