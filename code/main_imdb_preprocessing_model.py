#!/usr/bin/python3

import Utils
import Preprocessing
import TermFrequencyProcessing
import FeatureSelection
import pickle
import FileToReview
try:
   import _pickle as pickle
except:
   import pickle

"""
Input :
    pos_path : repertory positive directory
    neg_path : repertory negative directory
    selected_DB : which data we use ? imdb or the first one ? : Utils.DB_TWO or Utilis.DB_ONE
    is_bigrams : Do we use bigrams to compute tfidf etc ?
    k : percentage of most import word we keep (multual information)
    method : Mutual information or other method
    feature_space : if the feature has already been computed on an other dataset
Output :
    return the review preprocessed and with reduced vocabulary
    return fs : FeatureSelection Object
    return feature_space : if mutual information is choosen, feature_space contain the words we keep (only if feature_space is not given as input)
"""
def do_preprocessing(pos_path, neg_path, selected_DB, is_bigrams, k=None, method=None, features_space=None):
	f2r = FileToReview.FileToReview(pos_path, neg_path, selected_DB)
	pos_reviews, neg_reviews = f2r.buildReviewMatrix()

	# get a new instance for preprocessing
	# The new instance needs to know where positive and negative review directories are, also database no 
	prep = Preprocessing.Preprocessing(pos_path, neg_path, selected_DB, pos_reviews, neg_reviews, is_bigrams)

	# extract positive and negative vocabularies
	prep.extract_vocabulary()
	# print extracted vocabularies in dictionnary (json) format
	vocabs = prep.get_v()

	nb_neg_review = prep.get_nb_neg_review()
	nb_pos_review = prep.get_nb_pos_review()


    # get a new instance
    # The new instance needs to know where positive and negative review directories are, also database no
	tfp = TermFrequencyProcessing.TermFrequencyProcessing(pos_path, neg_path, selected_DB)
	tfp.compute_terms_frequency(vocabs)
	# print(tfp.get_overall_terms_frequency())
	# print(tfp.get_reviews_info())
	T = tfp.get_overall_terms_frequency()


	fs = FeatureSelection.FeatureSelection(T, nb_neg_review, nb_pos_review)

	if not features_space:
		features_space = fs.build_features_space(k, method)
		reduced_vocabs = fs.reduce_vocabs(vocabs, features_space)

		return vocabs, reduced_vocabs, fs, features_space

	reduced_vocabs = fs.reduce_vocabs(vocabs, features_space)
	return vocabs, reduced_vocabs, fs



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

#############################################################################################
# 1st use case: when necessary json files are not created yet
#############################################################################################

# MODEL FOR TRAINING_SET
k = 0.1 # % pourcentage of word we keep by mutual information
vocabs, reduced_vocabs, fs, features_space = do_preprocessing(pos_path_train, neg_path_train, selected_DB, is_bigrams, k, method)

pickle.dump(vocabs, open("../vocabs_train"+str(k)+".pickle", "wb"))
pickle.dump(reduced_vocabs, open("../reduced_vocabs_train"+str(k)+".pickle", "wb"))
pickle.dump(fs, open("../fs_train"+str(k)+".pickle", "wb"))
pickle.dump(features_space, open("../featurespace"+str(k)+".pickle", "wb"))

# vocabs = pickle.load(open("../vocabs_train"+str(k)+".pickle", "rb"))
# reduced_vocabs = pickle.load(open("../reduced_vocabs_train"+str(k)+".pickle", "rb"))
# fs = pickle.load(open("../fs_train"+str(k)+".pickle", "rb"))
# features_space = pickle.load(open("../featurespace"+str(k)+".pickle", "rb"))

print("debut tfidf train")
X_train_tfidf, Y = fs.create_bag_of_words_model(reduced_vocabs, features_space, vector_type=vector_type)
# print("debut doc2vec train")
# model_doc2tovec, X_train_doc2vec, Y = fs.create_doc2vec_model(vocabs, size=400)
# print("debut doc2vectfidf train")
# model_doc2tovec, X_train_doc2vec_tfidf, Y = fs.create_doc2vec_tfidf_model(vocabs, reduced_vocabs, features_space)

# We save as a pickle
# pickle.dump(X_train_doc2vec, open("../X_train_doc2vec.pickle", "wb"))
# pickle.dump(X_train_doc2vec_tfidf, open("../X_train_doc2vec_tfidf"+str(k)+".pickle", "wb"))
pickle.dump(X_train_tfidf, open("../X_train_tfidf"+str(k)+".pickle", "wb"))
pickle.dump(Y, open("../Y_train.pickle", "wb"))





# MODEL FOR TEST SET
print ("début preprocessing test")
vocabs, reduced_vocabs, fs = do_preprocessing(pos_path_test, neg_path_test, selected_DB, is_bigrams, features_space=features_space)

pickle.dump(vocabs, open("../vocabs_test"+str(k)+".pickle", "wb"))
pickle.dump(reduced_vocabs, open("../reduced_vocabs_test"+str(k)+".pickle", "wb"))
pickle.dump(fs, open("../fs_test"+str(k)+".pickle", "wb"))
pickle.dump(features_space, open("../featurespace"+str(k)+".pickle", "wb"))

# vocabs = pickle.load(open("../vocabs_test"+str(k)+".pickle", "rb"))
# reduced_vocabs = pickle.load(open("../reduced_vocabs_test"+str(k)+".pickle", "rb"))
# fs = pickle.load(open("../fs_test"+str(k)+".pickle", "rb"))
# features_space = pickle.load(open("../featurespace"+str(k)+".pickle", "rb"))


# print("debut doc2vec test")
# model_doc2tovec, X_test_doc2vec, Y_test = fs.create_doc2vec_model(vocabs, size=400)
# print("debut doc2vectfidf test")
# model_doc2tovec, X_test_doc2vec_tfidf, Y_test = fs.create_doc2vec_tfidf_model(vocabs,reduced_vocabs, features_space)
print("debut tfidf test")
X_test_tfidf, Y_test = fs.create_bag_of_words_model(reduced_vocabs, features_space, vector_type=vector_type)

# We save as a pickle
# pickle.dump(X_test_doc2vec, open("../X_test_doc2vec.pickle", "wb"))
pickle.dump(X_test_tfidf, open("../X_test_tfidf"+str(k)+".pickle", "wb"))
# pickle.dump(X_test_doc2vec_tfidf, open("../X_test_doc2vec_tfidf"+str(k)+".pickle", "wb"))
pickle.dump(Y_test, open("../Y_test.pickle", "wb"))

