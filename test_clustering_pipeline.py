#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on Dec 14, 2016

.. codeauthor: Svitlana <svitlana.vakulenko@gmail.com>
'''

import unittest

import numpy as np

from clustering_pipeline import hierarchical_clustering, run_pipeline
from tweet2vec.encode_char import generate_embeddings

MODELS_PATH = "models"
TEST_DATA_PATH = 'data/test_clustering/'


class TestClusteringPipeline(unittest.TestCase):
    def test_generate_embeddings(self):
        tweets_path = TEST_DATA_PATH + 'test.txt'
        embeddings = generate_embeddings([tweets_path, MODELS_PATH])
        assert embeddings

    def test_hierarchical_clustering(self):
        embs_path = TEST_DATA_PATH + 'test.npy'
        embeddings = np.load(embs_path)
        cluster_ids = hierarchical_clustering(embeddings)
        assert embeddings.shape[0] == len(cluster_ids)
        print cluster_ids

    def test_clustering_pipeline_cluster(self):
        tweets_path = TEST_DATA_PATH + 'test.txt'
        new_embs_path = TEST_DATA_PATH + 'generated_test.npy'
        results_path = TEST_DATA_PATH + 'results/'
        assert run_pipeline(tweets_path, MODELS_PATH, new_embs_path, do_generate=True,
                            show_results=True, output_path=results_path, distance_threshold=1.6)


if __name__ == '__main__':
    unittest.main()