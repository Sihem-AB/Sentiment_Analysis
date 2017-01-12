from gensim.models.doc2vec import TaggedDocument

class DocIterator(object):
    def __init__(self, doc_list, labels_list=None):
        if labels_list==None:
            labels_list = list(range(len(doc_list)))
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield TaggedDocument(words=doc, tags=[self.labels_list[idx]])