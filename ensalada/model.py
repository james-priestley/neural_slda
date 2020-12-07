import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from tomotopy import LLDAModel


def _counts_to_str(counts, min_length=0):
    documents = []
    n_words, n_docs = counts.shape
    for d in range(n_docs):
        doc = []
        for n in range(n_words):
            doc += [str(n)] * counts[n, d]

        if len(doc) > min_length:
            documents.append(doc)

    return documents


class LabeledLDA(TransformerMixin, BaseEstimator):

    """
    Labeled LDA via collapsed Gibbs sampling. This class provides a
    sklearn-esque wrapper for the `tomoto.LLDAModel` class.

    Parameters
    ----------
    n_components : int, optional
        Sets the number of corpus-wide topics. The final number of components
        will be equal to this number plus the number of unique labels passed
        during fitting.
    alpha :
    eta :
    burn_in :
    n_iter :
    kwargs :

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
    component_labels_  : list, length (n_components)
        Label for each component
    convergence_ : array, shape (n_iter)
    """

    def __init__(self, n_components=1, alpha=0.1, eta=0.01, burn_in=100,
                 n_iter=200, n_init=1, **kwargs):

        self.n_components = n_components
        self.alpha = alpha
        self.eta = eta
        self.burn_in = burn_in
        self.n_iter = n_iter
        self.n_init = n_init  # unused at the moment

        self._llda = LLDAModel(alpha=alpha, eta=eta, **kwargs)

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Array of document word counts
        y : list, length (n_samples)
            For each document, the list of labels
        """

        # convert X into documents for tomoto
        docs = _counts_to_str(X.T)

        # add corpus-wide topics
        background_topics = [
            f'background_{n}' for n in range(self.n_components)
        ]

        # add docs to model
        for d, l in zip(*[docs, y]):
            self._llda.add_doc(d, labels=l + background_topics)

        # run sampler
        self._llda.train(self.burn_in)
        self.perplexity_ = []
        for n in range(self.n_iter):
            self._llda.train(1)
            self.perplexity_.append(self._llda.perplexity)
        self.perplexity_ = np.array(self.perplexity_)

        # save model components
        self.components_ = np.stack(
            [self._llda.get_topic_word_dist(k)
             for k in range(self._llda.k)])
        self.n_components = self._llda.k
        self.component_labels_ = list(self._llda.topic_label_dict)

        return self

    def fit_transform(X, y):

        pass

    def perplexity(X):

        pass

    def score(X, y):

        pass

    def transform(X):

        pass

    def predict(X):

        pass
