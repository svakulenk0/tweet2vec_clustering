#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Created on Oct 6, 2016

.. codeauthor: svitlana vakulenko
    <svitlana.vakulenko@gmail.com>

Pipeline to create Tweet2Vec embeddings and cluster them

Dependencies:
* pandas
* fastcluster
* theano
* lasagne
'''
import os
import numpy as np
from collections import Counter
import csv

import fastcluster
from scipy.cluster.hierarchy import fcluster

from sklearn import preprocessing
from sklearn.metrics import pairwise

from tweet2vec.encode_char import save_embeddings


def hierarchical_clustering(embeddings, distance_metric='euclidean', max_d=1.0):
    '''
    Groups similar vector-embeddings into event-clusters
    '''
    # max_ds = [0.8, 0.9, 1.0]
    # precision flaw fix by cast to integers!!!
    a = np.int64(embeddings)
    # print 'Tweet vectors:', a.shape
    # check the number of unique embeddings
    b = np.ascontiguousarray(a).view(np.dtype((np.void,
                                     a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)

    unique_a = a[idx]
    print 'Unique tweet vectors:', unique_a.shape

    # check out the embeddings
    X = a.astype(float)
    # print X[0][0]

    X_scaled = preprocessing.scale(X)
    X_normalized = preprocessing.normalize(X_scaled, norm='l2')
    # print X_normalized.shape
    X = X_normalized
    HL = fastcluster.linkage(X, method='average', metric=distance_metric)
    print HL
    # for max_d in max_ds:
        # topics_out = embeddings + "_clusters_maxd_" + str(max_d) + ".csv"
        # # print topics_out
        # writer = csv.writer(open(topics_out, 'w'))

    print "max_d = ", max_d
    cluster_ids = fcluster(HL, max_d, criterion='distance')
    print len(cluster_ids)
    return cluster_ids


def get_clustered_tweet_ids(cluster_ids=None, n_topics=None):
    if cluster_ids == None:
        cluster_ids = cPickle.load(open('cluster_ids.p', 'rb'))
    freqTwCl = Counter(cluster_ids)
    n_clusters = len(freqTwCl)
    if not n_topics:
        n_topics = n_clusters
    print "n_clusters:", 
    print "Return top", n_topics
    npindL = np.array(cluster_ids)
    clusters = []
    for clfreq in freqTwCl.most_common(n_topics):
        cl = clfreq[0]
        freq = clfreq[1]
        clidx = (npindL == cl).nonzero()[0].tolist()
        # print len(clidx), "tweets"
        # print clidx
        clusters.append(clidx)
    return clusters


def show_tweets(clustered_tweets, tweets_path, output_path):
        # read tweets & write clusters
        with open(tweets_path, 'r') as f:
            # tweets = list(csv.reader(tsv, delimiter='\t'))
            tweets = f.read().splitlines()
        #     # array from 0 to 12999
            for idx, cluster in enumerate(clustered_tweets):
                print "Cluster (%d tweets)" % len(cluster)

        #     for (cl, freq, closest) in clusters:
        #         # print "freq:", freq
        #         # do not show duplicate tweets in the same cluster
                tweet_cluster = []
        #         tweet_indexes = []
        #         for tweet_index in closest:
        #             # print tweet_index
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                with open(output_path+'cluster'+str(idx), 'w+') as fout:
                    for tweet_index in cluster:
                        text = tweets[tweet_index].decode('utf-8')
            #             #.split('\t')[2].rstrip()
            #             if text not in tweet_cluster:
                        # print text
                        fout.write(text+'\n')


def run_pipeline(tweets_path, models_path, embs_path, do_generate=True, do_cluster=True,
                 show_results=False, output_path=None, distance_threshold=1.0):
    # 1st step: create vector-embeddings
    if do_generate:
        embeddings = save_embeddings(tweets_path, models_path, embs_path)
        assert embeddings
        embeddings = np.asarray(embeddings)
        # print embeddings
    # 2nd step: cluster
    if do_cluster or show_tweets:
        if not do_generate:
            embeddings = np.load(embs_path)
        cluster_ids = hierarchical_clustering(embeddings, max_d=distance_threshold)
        assert embeddings.shape[0] == len(cluster_ids)
        print cluster_ids
    if show_results:
        clustered_tweets = get_clustered_tweet_ids(cluster_ids)
        show_tweets(clustered_tweets, tweets_path, output_path)
    print "Finished."
    return True
