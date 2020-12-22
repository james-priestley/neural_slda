import argparse
import functools
import logging

import numpy as np
import torch
from torch import nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO
from pyro.optim import ClippedAdam

logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.INFO)

class ArgsObject:
    pass

class AmortizedLLDA:
    def __init__(self, num_words_per_doc=64, num_words=1024, num_docs=100, 
                 num_topics=6, batch_size=32, layer_sizes=(100,100), 
                 learning_rate=0.01, jit=True):
        self.args = ArgsObject()
        self.args.batch_size = batch_size
        self.args.layer_sizes = '-'.join([str(i) for i in layer_sizes])
        self.args.learning_rate = learning_rate
        
        # 
        self.args.num_words = num_words
        self.args.num_docs = num_docs
        self.args.num_words_per_doc = num_words_per_doc
        self.args.num_topics = num_topics
        
        self.args.jit = jit
    
    def fit(self, X, y):
        """
        
        """
        assert len(X) == len(y)
        self.args.num_docs, self.args.num_words_per_doc = X.shape
        self.args.num_topics = y.shape[1]
        
        data = torch.tensor(X.T)
        labels = torch.tensor(y)
        
        logging.info('-' * 40)
        logging.info('Training on {} documents'.format(self.args.num_docs))
        predictor = self._make_predictor()
        guide = functools.partial(self._parametrized_guide, predictor)
        Elbo = JitTraceEnum_ELBO if self.args.jit else TraceEnum_ELBO
        elbo = Elbo(max_plate_nesting=2)
        optim = ClippedAdam({'lr': self.args.learning_rate})
        svi = SVI(self._model, guide, optim, elbo)
        logging.info('Step\tLoss')
        for step in range(args.num_steps):
            loss = svi.step(data, labels, batch_size=self.args.batch_size)
            if step % 10 == 0:
                logging.info('{: >5d}\t{}'.format(step, loss))
        loss = elbo.loss(model, guide, data, labels, args=self.args)
        logging.info('final loss = {}'.format(loss))
        
    def sample(self):
        topic_weights, topic_words, data, labels = self._model()
        return topic_weights, topic_words, data.T, labels
    
    def _make_predictor(self):
        args = self.args
        layer_sizes = ([args.num_words + args.num_topics] +
                       [int(s) for s in args.layer_sizes.split('-')] +
                       [args.num_topics])
        logging.info('Creating MLP with sizes {}'.format(layer_sizes))
        layers = []
        for in_size, out_size in zip(layer_sizes, layer_sizes[1:]):
            layer = nn.Linear(in_size, out_size)
            layer.weight.data.normal_(0, 0.001)
            layer.bias.data.normal_(0, 0.001)
            layers.append(layer)
            layers.append(nn.Sigmoid())
        layers.append(nn.Softmax(dim=-1))
        return nn.Sequential(*layers)
    
    def _model(self, data=None, labels=None, batch_size=None):
        args = self.args
        # Globals.
        with pyro.plate("topics", args.num_topics):
            topic_weights = pyro.sample("topic_weights", dist.Gamma(1. / args.num_topics, 1.))
            topic_words = pyro.sample("topic_words",
                                      dist.Dirichlet(torch.ones(args.num_words) / args.num_words))
            label_prior = pyro.sample("label_prior", 
                                      dist.Beta(*torch.ones(2, args.num_topics)))

        # Locals.
        with pyro.plate("documents", args.num_docs) as ind:                                           
            if data is not None:
                with pyro.util.ignore_jit_warnings():
                    assert data.shape == (args.num_words_per_doc, args.num_docs)
                data = data[:, ind]

            if labels is not None:
                with pyro.util.ignore_jit_warnings():
                    assert labels.shape == (args.num_docs, args.num_topics)
                labels = labels[ind]

            labels = pyro.sample("labels", dist.Bernoulli(label_prior).to_event(1), 
                             obs=labels)
            auxiliary = pyro.sample("auxiliary", dist.Gamma(topic_weights, 1).to_event(1))

            doc_topics = labels*auxiliary
            doc_topics = pyro.sample("doc_topics", 
                                     dist.Delta(doc_topics / doc_topics.sum(axis=1)[...,None]).to_event(1))
            with pyro.plate("words", args.num_words_per_doc):
                word_topics = pyro.sample("word_topics", dist.Categorical(doc_topics),
                          infer={"enumerate": "parallel"})
                data = pyro.sample("doc_words", dist.Categorical(topic_words[word_topics]),
                                   obs=data)

        return topic_weights, topic_words, data, labels
    
    def _parametrized_guide(self, predictor, data, labels, batch_size=None):
        args = self.args
        # Use a conjugate guide for global variables.
        topic_weights_posterior = pyro.param(
                "topic_weights_posterior",
                lambda: torch.ones(args.num_topics),
                constraint=constraints.positive)
        topic_words_posterior = pyro.param(
                "topic_words_posterior",
                lambda: torch.ones(args.num_topics, args.num_words),
                constraint=constraints.greater_than(0.5))

        # Is this needed?
        label_posterior = pyro.param(
                "label_posterior", 
                lambda: torch.ones(2, args.num_topics), 
                constraint=constraints.positive)                              

        with pyro.plate("topics", args.num_topics):
            pyro.sample("topic_weights", dist.Gamma(topic_weights_posterior, 1.))
            pyro.sample("topic_words", dist.Dirichlet(topic_words_posterior))
            pyro.sample("label_prior", dist.Beta(*label_posterior))

        # Use an amortized guide for local variables.
        pyro.module("predictor", predictor)
        with pyro.plate("documents", args.num_docs, batch_size) as ind:
            data = data[:, ind]
            labels = labels[ind]
            counts = (torch.zeros(args.num_words, ind.size(0))
                           .scatter_add(0, data, torch.ones(data.shape)))

            augmented_input = torch.zeros(batch_size, args.num_words + args.num_topics)
            augmented_input[:, :args.num_words] = counts.transpose(0,1)
            augmented_input[:, args.num_words:] = labels

            doc_topics = predictor(augmented_input)
            pyro.sample("doc_topics", dist.Delta(doc_topics).to_event(1))


