import numpy as np
from sklearn import svm

class Retriever():
    def __init__(self, docs, embeddings):
        self.embeddings = embeddings
        self.docs = docs
        x = [doc_split.page_content for doc_split in docs]
        embeds = embeddings.embed_documents(x)
        embeds_np = np.array(embeds)
        embeds_np = embeds_np / np.sqrt((embeds_np**2).sum(1, keepdims=True)) # L2 normalize the rows
        self.embeds = embeds_np

    # This method is responsible for retrieving the top k most relevant documents
    # from a collection of documents based on a given query (question).
    def query(self, question, k=3): # k is the number of top results to return
        query = np.array(self.embeddings.embed_query(question))

        query = query / np.sqrt((query**2).sum()) # L2 normalize the query
        x = np.concatenate([[query], self.embeds])
        y = np.zeros(len(x))  # initialize labels
        y[0] = 1 # set the first element (the query) as positive

        # SVM training
        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
        clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=50000, tol=1e-5, C=1)
        clf.fit(x, y) # train

        similarities = clf.decision_function(x) # compute the similarity scores
        sorted_ix = np.argsort(-similarities)[1:] # sort in descending order and skip the first element (the query)
        res = []
        for i in sorted_ix[:k]:
            res.append(self.docs[i-1])
        return res