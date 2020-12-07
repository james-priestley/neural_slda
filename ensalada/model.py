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
    component_labels_  : array, length (n_components)
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

        Returns
        -------
        self
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
        self.component_labels_ = np.array(self._llda.topic_label_dict)

        return self

    def fit_transform(self, X, y):
        """
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Array of document word counts
        y : list, length (n_samples)
            For each document, the list of labels

        Returns
        -------
        doc_topic_dist : array, shape (n_samples, n_components)
            Inferred topic distribution for each sample
        """
        return self.fit(X, y).transform(X)

    def perplexity(self, X):

        pass

    def score(self, X):
        """
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Array of document word counts

        Returns
        -------
        logprob : array, shape (n_samples,)
        """
        return self._infer_topics(X)[1]

    def _infer_topics(self, X):
        # convert X into documents for tomoto, then cast as document class
        docs = _counts_to_str(X.T)
        tdocs = [self._llda.make_doc(d) for d in docs]

        # infer topic distributions
        doc_topic_dist, logprob = self._llda.infer(tdocs)
        doc_topic_dist = np.stack(doc_topic_dist)

        silent_bins = np.where(X.sum(axis=1) == 0)[0]
        if len(silent_bins):
            indices = silent_bins - np.arange(len(silent_bins))
            doc_topic_dist = np.insert(doc_topic_dist, indices, np.nan, axis=0)
            logprob = np.insert(logprob, indices, np.nan)

        return doc_topic_dist, logprob

    def transform(self, X):
        """
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Array of document word counts

        Returns
        -------
        doc_topic_dist : array, shape (n_samples, n_components)
            Inferred topic distribution for each sample
        """
        return self._infer_topics(X)[0]

    def predict(self, X, threshold=0.1):
        """This should predict the labels of each document by thresholding
        the posterior topic distributions.

        I suppose we could also create a null distribution somehow to threshold
        topic probabilities, maybe by just simulating the mean rate of each
        neuron.
        """

        y_hat = []
        doc_topic_dist = self.transform(X)
        for d in doc_topic_dist > 0.1:
            y_hat.append(self.component_labels_[d])

        return y_hat
