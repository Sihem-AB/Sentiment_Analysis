#!/usr/bin/python3

import Utils
import TermFrequencyProcessing
import operator
import gensim
import numpy as np
from math import floor
from math import log
from copy import deepcopy
from functools import reduce
from DocIterator import DocIterator
from nltk.corpus import sentiwordnet as swn


class FeatureSelection(object):
    def __init__(self, T, nb_neg_review, nb_pos_review):
        self.T = T  # overall terms frequency
        self.nb_neg_review = nb_neg_review
        self.nb_pos_review = nb_pos_review


    ###############################################################################
    # Bag of words model
    ###############################################################################

    """
		Let’s assume that after pre-processing of the data, we have N distinct words
		 in the entire document. Also, suppose that the document contains R reviews.
		Each review, is represented by a N -dimensional feature vector. The entries
		of these vectors correspond to the N words in the document and the feature values
		reflect the frequency of their corresponding words in the review. Alternatively,
		we can construct a binary feature vector where 1 indicates the presence of the
		corresponding word in the review and zero represents otherwise.
		Another option is also to use Tf-IDF value instead of binary value or frequency.
		===> Se 7th section in the article

		The goal of taking "features_space" in parameter is to take into account the order of features
			in Mutual Information for review vectors. Thus, each review vector is created according to that order.
			Example: features_space = ["hong", "kong", "monaco"], review1 = ["hong"], review2 = ["kong"].
				review_vector1 = [1, 0, 0], review_vector2 = [0, 1, 0]


		Input:
			vocabs: vocabulary object. It supposed to be reduced vocabulary according to feature selection
            features_space: selected features
            vector_type = one of these values: "FREQ", "TF-IDF", "BINARY", "TF-IDF-SENTIWORDNET"
	"""

    def create_bag_of_words_model(self, vocabs, features_space, vector_type="FREQ"):
        model = {}
        model[Utils.POS] = []
        model[Utils.NEG] = []

        nb_documents = len(vocabs[Utils.POS]["reviews"]) + len(vocabs[Utils.NEG]["reviews"])
        nb_features = len(features_space)
        if vector_type == "TF-IDF-SENTIWORDNET":
            nb_features += 2

        X = np.zeros((nb_documents, nb_features))
        Y = np.zeros(nb_documents)

        # features_space is not efficient to find the index of the term
        # features_space contains MI score for each term, and this is unnecessary for this task
        # replace MI score by index value so that we can acces the index of each term
        # indexed_features_space = deepcopy(features_space)
        indexed_features_space = {}
        index = 0
        for term, score in features_space.items():
            # print("term : ", term)
            indexed_features_space[term] = index
            index += 1

        for sentiment_class in [Utils.NEG, Utils.POS]:
            reviews = vocabs[sentiment_class]["reviews"]
            for review in reviews:
                id = review["id"]
                vec = self.create_review_vector(nb_features, indexed_features_space, review, vector_type)
                X[id, :] = vec
                Y[id] = sentiment_class
            # model[sentiment_class].append(vec)
        return X, Y

    # Retourne la valeur moyenne des negativite et de positivite d'un mot en le considerant lui meme et ces synonymes
    def get_average_word_pol(self, word):
        liste_syn = list(swn.senti_synsets(word))
        n = float(len(liste_syn))
        moy_pos = 0.0
        moy_neg = 0.0
        if n != 0:
            for elt in liste_syn:
                moy_pos += elt.pos_score()
                moy_neg += elt.neg_score()
            # elt.obj_score()
            moy_pos = moy_pos / n
            moy_neg = moy_neg / n
        return moy_pos, moy_neg

    def create_review_vector(self, nb_features, indexed_features_space, review, vector_type):
        # create zeros vector
        vec = np.zeros(nb_features)
        nb_word_in_review = review["nb_word"]
        sentences = review["sentences"]
        mean_pos = 0.0
        mean_neg = 0.0

        for sentence in sentences:
            for term, freq in sentence.items():
                try:
                    index = indexed_features_space[term]
                    if vector_type == "FREQ":
                        vec[index] = freq
                    elif vector_type == "TF-IDF" or vector_type == "TF-IDF-SENTIWORDNET":
                        vec[index] = self.compute_tf_idf(term, freq, nb_word_in_review)
                    else:  # BINARY
                        vec[index] = 1

                    if vector_type == "TF-IDF-SENTIWORDNET":
                        pos, neg = self.get_average_word_pol(term)
                        mean_pos += pos * freq
                        mean_neg += neg * freq

                except KeyError:
                    pass

        if vector_type == "TF-IDF-SENTIWORDNET":
            mean_pos = mean_pos / nb_word_in_review
            mean_neg = mean_neg / nb_word_in_review
            vec[-2] = mean_pos
            vec[-1] = mean_neg
        return vec


    def compute_tf_idf(self, term, freq, nb_word_in_review):
        # TODO In the article, the following formula is used: tf(w, d) = 1 + log(number of occurences of w in d)
        tf = freq / nb_word_in_review

        # idf(w, D) = log(card(D) / card(number of documents in which w occurs))
        tf_info = self.T[term]
        nb_pos_review_for_term = tf_info[Utils.POS]["nb_review"] if Utils.POS in tf_info.keys() else 0
        nb_neg_review_for_term = tf_info[Utils.NEG]["nb_review"] if Utils.NEG in tf_info.keys() else 0
        nb_review_for_term = nb_pos_review_for_term + nb_neg_review_for_term

        nb_review = self.nb_pos_review + self.nb_neg_review
        idf = log(nb_review / nb_review_for_term)

        return tf * idf

    ###############################################################################
    # Reduced Vocabs
    ###############################################################################

    """
		features_space will not probably contain all words in the corpus of review.
		This method removes words which are not	used in the features_space.

		Input:
			vocabs: extracted vocabulary by an instance of the class Preprocessing
			features_space: feature space computed by feature utilty method like Mutual Information
	"""

    def reduce_vocabs(self, vocabs, features_space):
        reduced_vocabs = deepcopy(vocabs)  # create a new dict object without linking to the old one

        # remove terms which are not representative according to the feature utility method
        for sentiment_class in [Utils.NEG, Utils.POS]:
            reviews = reduced_vocabs[sentiment_class]["reviews"]

            for review in reviews:
                self.reduce_review(review, features_space)

            # update reviews
            reduced_vocabs[sentiment_class]["reviews"] = reviews

        return reduced_vocabs

    """
		features_space will not probably contain all words in the corpus of review.
		This method updates each sentence of the given review (i.e. list words with their frequency) by removing words which are not used in the features_space
	"""

    def reduce_review(self, review, features_space):
        sentences = review["sentences"]

        for sentence in sentences:
            terms_to_be_removed = []

            for term, freq in sentence.items():
                if term not in features_space:
                    terms_to_be_removed.append(term)

        for term in terms_to_be_removed:
            del sentence[term]

        sentences_ordered = review["sentences_ordered"]
        new_sentences_ordered = []

        for sentence_ordered in sentences_ordered:

            new_sentence = []
            for word in sentence_ordered:
                if word not in terms_to_be_removed:
                    # if word in features_space:
                    new_sentence.append(word)

            new_sentences_ordered.append(new_sentence)

        # Update
        review["sentences_ordered"] = new_sentences_ordered

    ###############################################################################
    # Mutual Information
    ###############################################################################

    """
		Input:
			k : 	pourcentage of words with largest value.
					Values are computed through feature utility method.
					It should be between 0 and 1. ==> Float
			method: method name. ==> String.
					By now, there is only 1 feature utility method which is "MI" (Mutual Information)

		Returns k% terms with largest values


		Compute feature utility.
        According to the source below, there are 3 feature utility functions: mutual information, chi square test and frequency
        source: http://nlp.stanford.edu/IR-book/html/htmledition/feature-selection-1.html#fig:featselalg
        For now, there is only 1 available feature utility: Mutual Information (MI)
	"""

    def build_features_space(self, k=1, method="MI"):
        if k > 1 or k <= 0:
            k = 1  # default value

        FU = None  # Feature Utility
        if method == "MI":
            FU = self.compute_MI()

        nb_features = len(FU)
        # transform pourcentage into number
        top_k = floor(k * nb_features)  # i.e floor(30.2) = 30
        print("MI: The number of terms with largest values according to the parameter 'k' ====>", top_k, "\n")

        # sort list by mutual information value in descending order
        # extract top k terms
        k_repr_terms = sorted(FU, key=lambda x: (x[1], x[0]), reverse=True)[:top_k]
        # print(k_repr_terms)

        # return a dict object whose key is a term and value is its MI score
        k_repr_terms = Utils.make_dict_from_two_value_paired_list(k_repr_terms)
        # the dict object is not sorted by MI score # TODO is it important?
        return k_repr_terms

    def compute_MI(self):
        """
			conditional mutual information and dealing with zero probabilities. For this purpose, Utils.flexible_log() is created
			source: http://stats.stackexchange.com/questions/73502/conditional-mutual-information-and-how-to-deal-with-zero-probabilities

			formula source: http://nlp.stanford.edu/IR-book/html/htmledition/mutual-information-1.html


			N_11: number of positive review that contains the word
			N_10: number of negative review that contains the word
			N_01: number of positive review that doesn't contain the word
			N_00: number of negative review that doesn't contain the word

			N = N_11 + N_10 + N_01 + N_00
			N_1. = N_10 + N_11
			N_0. = N_00 + N_01
			N_.1 = N_01 + N_11
			N_.0 = N_10 + N_00

			formula:
				I(U;C) = (N_11/N * log_2(N*N_11/N_1.*N_.1) ) +
						 (N_01/N * log_2(N*N_01/N_0.*N_.1) ) +
						 (N_10/N * log_2(N*N_10/N_1.*N_.0) ) +
						 (N_00/N * log_2(N*N_00/N_0.*N_.0) )
		"""

        L = []
        for term, tf_val in self.T.items():
            N_11 = tf_val[Utils.POS]["nb_review"] if Utils.POS in tf_val.keys() else 0
            N_10 = tf_val[Utils.NEG]["nb_review"] if Utils.NEG in tf_val.keys() else 0
            N_01 = self.nb_pos_review - N_11
            N_00 = self.nb_neg_review - N_10
            N__1 = N_01 + N_11
            N_1_ = N_10 + N_11
            N__0 = N_10 + N_00
            N_0_ = N_00 + N_01
            N = N_11 + N_10 + N_01 + N_00

            # TODO break a line for the formula below
            result = ((N_11 / N) * Utils.flexible_log((N * N_11) / (N_1_ * N__1))) + (
            (N_01 / N) * Utils.flexible_log((N * N_01) / (N_0_ * N__1))) + (
                     (N_10 / N) * Utils.flexible_log((N * N_10) / (N_1_ * N__0))) + (
                     (N_00 / N) * Utils.flexible_log((N * N_00) / (N_0_ * N__0)))

            L.append((term, result))

        return L

    ###############################################################################
    # Doc2Vec model
    ###############################################################################


    # bag-of-words features haver two major weaknesses: they lose the ordering of the words and they also ignore semantics of the words.
    # That's why we use Doc2Vec. We'll get a representation of the documents who take the order of the words into account

    """
		Input:
			vocabs: vocabulary object. It supposed to be a reduced vocabulary
			size : size of the vectors who will represents the documents
			nb_epoch : number of epoch doc2Vec will do in order to learn the vectors
			learning_rate : wich rate we want to learn

		Output:
			return X, Y
			X : ndarray of size (nbdocuments, size)
			Y : ndarray of size (nbdocuments)

	"""

    def create_doc2vec_model(self, vocabs, size=300, nb_epochs=10, learning_rate=0.025):
        # model = {Utils.POS: [], Utils.NEG: []

        nb_documents = len(vocabs[Utils.POS]["reviews"]) + len(vocabs[Utils.NEG]["reviews"])
        X = np.zeros((nb_documents, size))
        Y = np.zeros(nb_documents)

        # We get all the documents and the ids of each document
        for sentiment_class in [Utils.NEG, Utils.POS]:
            reviews = vocabs[sentiment_class]["reviews"]
            docs = []
            ids = []
            for review in reviews:
                sentences = review["sentences_ordered"]
                # We don't need to have the concept of sentences. We use reduce to flat the list in order to only have the order.
                docs.append(reduce(operator.add, sentences))
                ids.append(review["id"])

        # TRAINING DOC2VEC
        # doc2vec need an iterator
        it = DocIterator(docs, labels_list=ids)
        model = gensim.models.Doc2Vec(size=size, window=10, min_count=1, workers=11, alpha=learning_rate,
                                      min_alpha=0.025)  # use fixed learning rate
        model.build_vocab(it)
        for epoch in range(nb_epochs):
            model.train(it)
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no deca
            model.train(it)

        # We transform the model to have X and Y
        for sentiment_class in [Utils.NEG, Utils.POS]:
            reviews = vocabs[sentiment_class]["reviews"]
            for review in reviews:
                X[review["id"], :] = model.docvecs[review["id"]]
                Y[review["id"]] = sentiment_class
        return model, X, Y

    def create_doc2vec_tfidf_model(self, vocabs, reduced_vocabs, features_space, size=300, nb_epochs=10,
                                   learning_rate=0.025):
        model, X_docvecs, Y = self.create_doc2vec_model(vocabs, size, nb_epochs, learning_rate)
        Xtfidf, Y = self.create_bag_of_words_model(reduced_vocabs, features_space, "TF-IDF")

        return model, np.concatenate((X_docvecs, Xtfidf), axis=1), Y