#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Sep 28, 2016

.. codeauthor: svitlana vakulenko <svitlana.vakulenko@gmail.com>

Evaluation of tweet clustering performance
'''

import numpy as np
import fastcluster
from scipy.cluster.hierarchy import cophenet, dendrogram
from scipy.spatial.distance import pdist
from scipy.optimize import basinhopping, minimize, fmin, fmin_powell
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
from collections import defaultdict
from sklearn.metrics.pairwise import pairwise_distances
import csv,codecs,cStringIO
from sklearn.metrics import  pairwise
from collections import Counter
from sklearn.metrics import pairwise_distances_argmin_min, adjusted_mutual_info_score
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.cluster import MeanShift
import pandas
from sklearn.manifold import TSNE
from sklearn import preprocessing
import feather
from scipy.sparse import lil_matrix
import seaborn as sns

from assess_against_gs import get_gs_pointers, get_labels_true

ifrim_file = 'embeddings_ifrim.feather'
# tweet2vec_file = 'embeddings_tweet2vec.feather'
# tweet2vec_file = 'embeddings_tweet2vec_wo_users.feather'
tweet2vec_file = 'embeddings_tweet2vec_wo_users_wo_urls.feather'
# tweet2vec_file = 'embeddings_tweet2vec_wo_users_wo_urls2.feather'
# tweet2vec_file = 'embeddings_tweet2vec_wo_users_wo_urls2.npy'


def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


def plot_PCA(data):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    x = []
    y = []
    z = []
    for item in data:
        x.append(item[0])
        y.append(item[1])
        z.append(item[2])
    plt.close('all') # close all latent plotting windows
    fig1 = plt.figure() # Make a plotting figure
    ax = Axes3D(fig1) # use the plotting figure to create a Axis3D object.
    pltData = [x,y,z]
    ax.scatter(pltData[0], pltData[1], pltData[2], 'bo') # make a scatter plot of blue dots from the data
    # make simple, bare axis lines through space:
    xAxisLine = ((min(pltData[0]), max(pltData[0])), (0, 0), (0,0)) # 2 points make the x-axis line at the data extrema along x-axis
    ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r') # make a red line for the x-axis.
    yAxisLine = ((0, 0), (min(pltData[1]), max(pltData[1])), (0,0)) # 2 points make the y-axis line at the data extrema along y-axis
    ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r') # make a red line for the y-axis.
    zAxisLine = ((0, 0), (0,0), (min(pltData[2]), max(pltData[2]))) # 2 points make the z-axis line at the data extrema along z-axis
    ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r') # make a red line for the z-axis.
    # label the axes
    ax.set_xlabel("x-axis label")
    ax.set_ylabel("y-axis label")
    ax.set_zlabel("y-axis label")
    ax.set_title("The title of the plot")
    plt.show() # show the plot


def reduce_dimensionality_tsne(embeddings='../embeddings_test_snow.npy',
                          dimensions=2):
    # e - np array
    # t-stochastic neighbor embedding (https://lvdmaaten.github.io/tsne/) TSNE
    with open(embeddings, 'r') as f:
        e = np.load(f).astype('float64')
        print "\nComputing t-SNE reduction of vectors to {}D".format(dimensions)
        tsne_model = TSNE(n_components=dimensions, n_iter=10000000, metric="correlation", learning_rate=50, early_exaggeration=500.0, perplexity=40.0)  # random_state=0
        np.set_printoptions(suppress=True)
        vectors = tsne_model.fit_transform(e)
        return vectors

def scatter(x):
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # We add the labels for each digit.
    # txts = []
    # for i in range(10):
    #     # Position of each label.
    #     xtext, ytext = np.median(x[colors == i, :], axis=0)
    #     txt = ax.text(xtext, ytext, str(i), fontsize=24)
    #     txt.set_path_effects([
    #         PathEffects.Stroke(linewidth=5, foreground="w"),
    #         PathEffects.Normal()])
    #     txts.append(txt)

    return f, ax, sc


def reduce_dimensionality_pca(embeddings='../embeddings_test_snow.npy'):
    from sklearn.decomposition import PCA
    # if n_components == ‘mle’, Minka’s MLE is used to guess the dimension
    # if n_components is not set all components are kept
    with open(embeddings, 'r') as f:
        e = np.load(f)
        # pca = PCA(n_components=2)
        # pca.fit(e)
        # print pca.explained_variance_ratio_
        # from matplotlib.mlab import PCA
        #construct your numpy array of data
        pca = PCA(n_components='mle')
        pca.fit(e)

        #this will return an array of variance percentages for each component
        # results.fracs

        #this will return a 2d array of the data projected into PCA space
        # plot_PCA(results.Y)


def inconsist(Z, depth):
    from scipy.cluster.hierarchy import inconsistent

    incons = inconsistent(Z, depth) # optional , depth)
    # link statistics
    # print incons[-40:][5]  # [-10:]
    # print max(incons[:][5])  # [-10:]
    # print min(incons[:][5])  # [-10:]
    # print incons#[:][5].transpose()
    print incons[:,3]#.transpose()
    print max(incons[:,3])#.transpose()
    print min(incons[:,3])#.transpose()
    # draw_histogram(incons[:,3].transpose())


def draw_histogram(a):
    import matplotlib.pyplot as plt
    plt.hist(a)  # plt.hist passes it's arguments to np.histogram
    # >>> plt.title("Histogram with 'auto' bins")
    plt.show()


def elbow(Z):
    last = Z[-500:, 2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)
    plt.plot(idxs, last_rev)

    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]
    plt.plot(idxs[:-2] + 1, acceleration_rev)
    plt.show()
    k = acceleration_rev.argmax() + 2
    # if idx 0 is the max of this we want 2 clusters
    print "clusters:", k


def run_MeanShift(X, labels):
        # alternative clustering method TOO SLOW
    ms = MeanShift()
    ms.fit(X)
    cluster_centers = ms.cluster_centers_
    print(cluster_centers)
    n_clusters_ = len(np.unique(labels))
    print("Number of estimated clusters:", n_clusters_)


def compare_clusters(labels_1, labels_2):
    return adjusted_mutual_info_score(labels_1, labels_2)


def hierarchical_clustering(interval, embeddings, distance_metric,
                            n_clusters=None, max_ds=None, infile=None,
                            writer=None,
                            calc_v_measure=False, show_topics=False,
                            silhouette=False, numpy=False):
    ## Load ifrim tweet representation matrix from file

    # with open(embeddings+ifrim_file, 'r') as f:

    # show what we are processing
    print infile
    # get pandas.DataFrame
    if infile == 'ifrim':
        e = feather.read_dataframe(embeddings + ifrim_file)
    else:
        # print embeddings + tweet2vec_file
        if numpy:
            a = np.load(embeddings + tweet2vec_file)

        else:
            e = feather.read_dataframe(embeddings + tweet2vec_file)
            print e.shape
            a = e.as_matrix()
            print a[0][0]
            # print e[0]
            print e.drop_duplicates(keep='last').shape
            e = e.round(decimals=9)
            print e.drop_duplicates(keep='last').shape
            # return
            # precision flaw!!!
            # a = np.int64(e.as_matrix())
            a = e.as_matrix()
    #         # a = numpy.around(a, decimals=5)
            print a[0][0]
            return

    # print a.shape

    # # check the number of unique embeddings
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)

    unique_a = a[idx]
    print unique_a.shape
    # X = a


    # check out the embeddings
    # print e
    # X_scaled = preprocessing.scale(lil_matrix(e).todense())
    X = a.astype(float)
    print X[0][0]
    return


    X_scaled = preprocessing.scale(X)
    # print "X_scaled.shape:", X_scaled.shape
    # cut a small sample of rows to test
    X_normalized = preprocessing.normalize(X_scaled, norm='l2')#[:12000]
    # print "X_normalized.shape:", X_normalized.shape
    print X_normalized.shape
    X = X_normalized
    # print np.isnan(X).any()
    # print np.where(X == 0)
    # check out the normalized matrix
    # print X
    # print np.iszero(X)
    # X = pairwise_distances(X, metric='cosine')  # cityblock chebychev euclidean
    # choose linkage method and metric
    # distMatrix = pairwise_distances(X, metric=distance_metric)
    HL = fastcluster.linkage(X, method='average', metric=distance_metric)  # average ward single complete weighted centroid median
    # HL = fastcluster.linkage(distMatrix, method='average')
    # metric= euclidean sqeuclidean seuclidean mahalanobis cityblock chebychev minkowski cosine correlation canberra braycurtis hamming jaccard
    #  average trained on snow 18: 0.835892520027 ward 0.6

    ## find_optimal_number_of_clusters
    # find_optimal_number_of_clusters_auto(X, HL, distance_metric)
    # max_d = find_optimal_distance_threshold(HL, X, distance_metric, x0=[10])


    ## report clusters
    # n_clusters = 3224
# 20 36
    # max_d = 10  # 24 max_d as in max_distance 500 20 25 None 10
    # max_ds = [10]  # 24 max_d as in max_distance 500 20 25 None
    metrics = []
    for max_d in max_ds:

        if not writer and show_topics:
            topics_out = embeddings + infile + "_clusters_maxd_"+str(max_d)+".csv"
            print topics_out
            writer = csv.writer(open(topics_out, 'w'))

    #     evaluate_clusters(HL, X, n_clusters, embeddings, distance_metric, max_d)
        metric = report_clusters(interval, HL, X, embeddings, distance_metric, infile, max_d, n_clusters,
                    writer=writer, v_measure=calc_v_measure, show_topics=show_topics, calculate_silhouette=silhouette)
        metrics.append(metric)
        # writer.writerow([''])

    if silhouette:
        print "Maximum Silhuette: ", max(metrics)
    else:
        print "Maximum v_measure: ", max(metrics)


# def estimate_silhouette(n_clusters, HL, X):
#     print n_clusters
#     cluster_ids = fcluster(HL, t=n_clusters, criterion='maxclust')
#     # score = metrics.silhouette_score(distance_matrix, clusters, metric='precomputed')
#     silhouette_avg = silhouette_score(X, cluster_ids)  # metric euclidean default TODO pass distance_metric
#     print silhouette_avg
#     return -1*silhouette_avg


            # c, coph_dists = cophenet(HL, pdist(e))
            # print c
        # average



        ## choose number of clusters
            # set cut-off
        # max_d = 28  # 24 max_d as in max_distance
            # TODO calculate cut-off
        # elbow(HL)


            # plot the dendrogram truncated with the cut-off value
                # lastnclusters = 120
                # fancy_dendrogram(
                #     HL,
                #     truncate_mode='lastp',
                #     p=lastnclusters,
                #     leaf_rotation=90.,
                #     leaf_font_size=12.,
                #     show_contracted=True,
                #     annotate_above=10,
                #     max_d=max_d,  # plot a horizontal cut-off line
                # )
                # plt.show()


def plot_counter(counter):
    labels, values = zip(*counter.most_common())
    print values
    df = pandas.DataFrame.from_items(values)
    print df
    print df.quantile(q=0.2)

    print np.percentile(values, 50)
    # indexes = np.arange(len(labels))
    # width = 1
    # plt.bar(indexes, values, width)
    # plt.xticks(indexes + width * 0.5, labels)
    # plt.show()


def estimate_silhouette(parameter, HL, X, distance_metric):
    print "max_d = ", parameter
    cluster_ids = fcluster(HL, parameter, criterion='distance')

    freqTwCl = Counter(cluster_ids)
    print "n_clusters:", len(freqTwCl)
    # print "n_clusters =", n_clusters
    # cluster_ids = fcluster(HL, t=n_clusters, criterion='maxclust')
    # score = metrics.silhouette_score(distance_matrix, clusters, metric='precomputed')
    silhouette_avg = silhouette_score(X, cluster_ids, metric=distance_metric)  # metric euclidean default TODO pass distance_metric
    print "The average silhouette_score is :", silhouette_avg
    return -1*silhouette_avg



# def search_local_max_interval(X, HL, cur_x, cur_y):
#     # estimate next neighbour forward
#     next_x = cur_x + 1
#     next_y = estimate_silhouette(next_x, HL, X)
#     # estimate previous neighbour backward
#     prev_x = cur_x - 1
#     prev_y = estimate_silhouette(prev_x, HL, X)
#     max_indx = np.argmax([prev_y, cur_y, next_y])
#     if max_indx == 1:
#         print "Local max: ", prev_x
#     elif max_indx == 0: # search backward
#         step = -1
#         climb(prev_x, step, HL, X, cur_y, step)
#     else:  # search forward
#         step = 1
#         climb(next_x, step, HL, X, cur_y, step)


def climb(cur_x, HL, X, cur_y, distance_metric, step):
    new_x = cur_x + step
    new_y = estimate_silhouette(new_x, HL, X, distance_metric)
    if new_y > cur_y: # continue climbing
        climb(new_x, HL, X, new_y, distance_metric, step)
    else:
        print "Local max: ", cur_x
        estimate_dunn(cur_x, HL, X, distance_metric)
        print '\n'


def find_optimal_number_of_clusters_auto(X, HL, distance_metric, n0=2, step=3, max_n=50):
    # range_n_clusters = list(range(2,50))  # MAX 12039 and MIN # clusters
    # initial status
    n_clusters = n0  # MAX 12039 and MIN # clusters
    prev_f = 0
    direction = 'init' # skip the first hill

    # print range_n_clusters
    # for n_clusters in range_n_clusters:
    # TODO loop here
    while n_clusters < max_n:
        cluster_ids = fcluster(HL, t=n_clusters, criterion='maxclust')
        print "n_clusters =", n_clusters

        silhouette_avg = silhouette_score(X, cluster_ids, metric=distance_metric)
        print "The average silhouette_score is :", silhouette_avg
        print '\n'


        if silhouette_avg < prev_f: # new direction desc
            if direction == 'asc':  # detected local maximum point
                # print "Local max: ", prev_x
                if prev_x != n0:  # skip the first hill
                    # search_local_max_interval(X, HL, prev_x, silhouette_avg)
                    # go backwards
                    climb(n_clusters, HL, X, silhouette_avg, distance_metric, step=-1)
            direction = 'desc'
        else:  # new direction asc
            # if direction == 'desc':  # detected local minimum point
            direction = 'asc'

        # save to track back
        prev_f = silhouette_avg
        prev_x = n_clusters

        # get new x point
        n_clusters += step


def find_optimal_distance_threshold(HL, X, distance_metric, x0=[10]):
    # minimizer_kwargs = {"args": (HL, X, metrics)}
    # 4 optimization algs
    # ret = basinhopping(silhouette_score, x0, stepsize=1, minimizer_kwargs=minimizer_kwargs, niter=200)
    ret = minimize(estimate_silhouette, x0, args=(HL, X, distance_metric), method='Nelder-Mead', options={'disp': True, 'xtol': 1})  # Powell Nelder-Mead 'ftol': 0.1,
    # ret = fmin(estimate_silhouette, x0, args=(HL, X), disp=1)
    print ret
    # ret = fmin_powell(estimate_silhouette, x0, args=params)
    # print("global minimum: x = %.4f, f(x0) = %.4f" % (ret.xopt, ret.fopt))
    print("global minimum: x = %.4f, f(x0) = %.4f" % (ret.x, ret.fun))
    return ret.x


    # x0 = [2]
    # # minimizer_kwargs = {"args": (HL, X, distance_metric,)}
    # # ret = basinhopping(silhouette_score, x0, stepsize=1, minimizer_kwargs=minimizer_kwargs, niter=200)
    # params = (HL, X)
    # # ret = minimize(silhouette_score, x0, args=(X,))
    # ret = fmin(silhouette_score, x0, args=params)
    # print("global minimum: x = %.4f, f(x0) = %.4f" % (ret.x, ret.fun))



    # ca = KMeans()
    # param_grid = {"n_clusters": range(2, 11)}

    # # run randomized search
    # search = GridSearchCV(
    #     ca,
    #     param_distributions=param_dist,
    #     n_iter=n_iter_search,
    #     scoring=silhouette_score,
    #     cv= # can I pass something here to only use a single fold?
    #     )
    # search.fit(distance_matrix)


def find_optimal_number_of_clusters_man():
        range_n_clusters = [38,37,36, 5000]  # MAX 12039 and MIN # clusters

        for n_clusters in range_n_clusters:
        # range_threshold = [2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.3, 3.4]
        # range_threshold = [2.6, 2.8, 3, 3.2, 3.4]
            # depth = 5
            # inconsist(HL, depth)

            # range_threshold = [3.4]
            # for threshold in range_threshold:


            # # cluster_ids = fcluster(HL, max_d, criterion='distance')
            #     print "Inconsistency threshold: " + str(threshold)
                # cluster_ids = fcluster(HL, t=threshold, criterion='inconsistent', depth=depth)
                # freqTwCl = Counter(cluster_ids)
                # n_clusters = len(freqTwCl)
            # cut by number of clusters
            cluster_ids = fcluster(HL, t=n_clusters, criterion='maxclust')
            dunn = dunn_fast(X, cluster_ids)
            print("For n_clusters =", n_clusters,
                      "The Dunn index is :", dunn)
            silhouette_avg = silhouette_score(X, cluster_ids)
            print("For n_clusters =", n_clusters,
                      "The average silhouette_score is :", silhouette_avg)


def evaluate_clusters(HL, X, n_clusters, embeddings, distance_metric, max_d):
    # cluster_ids = fcluster(HL, t=n_clusters, criterion='maxclust')

     # show full dendrogram to check distance threshold manually
        # plt.figure(figsize=(25, 10))
        # plt.title('Hierarchical Clustering Dendrogram')
        # plt.xlabel('sample index')
        # plt.ylabel('distance')
        # dendrogram(
        #     HL,
        #     leaf_rotation=90.,  # rotates the x axis labels
        #     leaf_font_size=8.,  # font size for the x axis labels
        # )
        # plt.show()

    print "max_d = ", max_d
    cluster_ids = fcluster(HL, max_d, criterion='distance')

    freqTwCl = Counter(cluster_ids)
    print "n_clusters:", len(freqTwCl)

    # dunn = dunn_fast(X, cluster_ids, distance_metric)
    # print "The Dunn index is :", dunn
    silhouette_avg = silhouette_score(X, cluster_ids)
    print "The average silhouette_score is :", silhouette_avg
    print '\n'


def report_clusters(interval, HL, X, embeddings, distance_metric, infile,
                    max_d=None,
                    n_clusters=None, show_topics=True,
                    writer=None, calculate_silhouette=False,
                    plot_diagram=False, v_measure=False):

    # show full dendrogram to check distance threshold manually
    if plot_diagram:
        plt.figure(figsize=(25, 10))
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        dendrogram(
            HL,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8.,  # font size for the x axis labels
        )
        plt.show()

    # max_ds = [10, 9.5, 10.5]  # 24 max_d as in max_distance 500 20 25
    # for max_d in max_ds:
    # if max_d:
    print "max_d = ", max_d
    cluster_ids = fcluster(HL, max_d, criterion='distance')

    # elif n_clusters:
    #     # n_clusters = 20
    #     cluster_ids = fcluster(HL, t=n_clusters, criterion='maxclust')
        # print compare_clusters(cluster_ids, cluster_ids)

    # print cluster_ids
    freqTwCl = Counter(cluster_ids)
    print "n_clusters:", len(freqTwCl)

    if v_measure:
        return compare_with_gs(interval, cluster_ids)



            # clusters = defaultdict(list)
            # for i, cluster in enumerate(cluster_ids):
            #     clusters[cluster].append(i)
            # # print clusters


            # print len(clusters) # number of clusters
            # plt.figure(figsize=(10, 8))
            # plt.scatter(X[:,0], X[:,1], c=clusters, cmap='prism')  # plot points with cluster dependent colors
            # plt.show()

#             print "hclust cut threshold:", dt
# #               indL = sch.fcluster(L, dt, 'distance')
#             indL = sch.fcluster(L, dt*distMatrix.max(), 'distance')
        #print "indL:", indL


    # df.plot(kind='bar')
    # plot_counter(freqTwCl)


    # print(freqTwCl)
    if calculate_silhouette:
        silhouette_avg = silhouette_score(X, cluster_ids)
        print "Average silhouette_score:", silhouette_avg
        return silhouette_avg

    print '\n'

    if show_topics:
        npindL = np.array(cluster_ids)
    #               print "top50 most populated clusters, down to size", max(10, int(X.shape[0]*0.0025))
        # freq_th = max(10, int(X.shape[0]*0.0025))
        clusters = []
        n_topics = 20
        for clfreq in freqTwCl.most_common(n_topics):
            cl = clfreq[0]
            freq = clfreq[1]
            ## threshold frequency
            # if freq >= freq_th:
                #print "\n(cluster, freq):", clfreq
            clidx = (npindL == cl).nonzero()[0].tolist()
            first = clidx[0]
            last = clidx[-1]
            # print last
            # ids from 0 to 12999

            cluster_centroid = X[clidx].mean(axis=0)
            # print len(cluster_centroid)

    # print cluster_centroids.shape
            # find a single tweet-medoid
            # closest, _ = pairwise_distances_argmin_min(cluster_centroid, X, metric='cityblock')
            dist = pairwise.pairwise_distances(cluster_centroid, X, metric=distance_metric)
            # closest = np.argmin(dist)
            closest = np.argsort(dist)[0][:3]
            # prepand the first tweet of the cluster
            closest = np.insert(closest, 0, first)
            closest = np.append(closest, last)
            print closest
            # np.argsort(dist)[:3]
            # print np.argmin(dist)
            # print 'id:',closest, 'dist:',min(dist)

            clusters.append((cl, freq, closest))

        # read tweets & write clusters
        with open(embeddings + '/prepared_tweet2vec_wo_users_wo_urls.txt', 'r') as f:
            # tweets = list(csv.reader(tsv, delimiter='\t'))
            tweets = f.read().splitlines()
            # array from 0 to 12999

            for (cl, freq, closest) in clusters:
                # print "freq:", freq
                # do not show duplicate tweets in the same cluster
                tweet_cluster = []
                tweet_indexes = []
                for tweet_index in closest:
                    # print tweet_index
                    text = tweets[tweet_index].decode('utf-8')#.split('\t')[2].rstrip()
                    if text not in tweet_cluster:
                        # print text
                        tweet_cluster.append(text)
                        tweet_indexes.append(tweet_index)
                # print '\n'
                # report only non-trivial cases
                if writer and len(tweet_cluster)>1:
                    print tweet_indexes
                    writer.writerow([infile, str(freq), '\n'.join(tweet_cluster)])

        # closest, _ = pairwise_distances_argmin_min(cluster_centroids, X)
        # print closest


def load_embeddings(embeddings):
    with open(embeddings + '.npy', 'r') as f:
            e = np.load(f)
            print e.shape

# TODO pass which corpus gs to load
def compare_with_gs(interval, all_labels_pred, corpus=None):
    # print all_labels_pred
    labels_pred = []
    for idx in get_gs_pointers(interval):
        labels_pred.append(all_labels_pred[idx])
    # print labels_pred
    # print labels_true
    labels_true = get_labels_true(interval)
    assert len(labels_pred) == len(labels_true)

    print 'normalized_mutual_info_score: ', normalized_mutual_info_score(labels_true, labels_pred)
    print 'adjusted_mutual_info_score: ', adjusted_mutual_info_score(labels_true, labels_pred)
    print 'adjusted_rand_score: ', adjusted_rand_score(labels_true, labels_pred)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(labels_true, labels_pred)
    print 'homogeneity: ', homogeneity
    print 'completeness: ', completeness
    print 'v_measure: ', v_measure

    return v_measure
    # print '\n'

class UnicodeWriter:
    def __init__(self, f, dialect=csv.excel, encoding="utf-8-sig", **kwds):
        self.queue = cStringIO.StringIO()
        self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        '''writerow(unicode) -> None
        This function takes a Unicode string and encodes it to the output.
        '''
        self.writer.writerow([s.encode("utf-8") for s in row])
        data = self.queue.getvalue()
        data = data.decode("utf-8")
        data = self.encoder.encode(data)
        self.stream.write(data)
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


def run_evaluation():
    '''
    Main method for clustering evaluation
    '''
    # produce clusterings
    # intervals = ['18:00', '22:00', '23:15', '01:00', '01:30']
    # intervals = ['22:00', '23:15', '01:00', '01:30']
    intervals = ['01:30']
    # all_d_s = [28, 25, 22, 21, 20, 19.5, 19.4, 19.35, 19.3, 19.25, 19.2, 19.1, 19, 18.8, 18.5, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 5, 4, 3, 2, 1.5, 1.2, 1.1, 1, 0.9, 0.8, 0.5, 0.2, 0.1, 0.01]
    # d_s = all_d_s
    # d_s_ifrim = [0.1, 0.01]
    # d_s = d_s_ifrim
    all_d_s_tweet2vec = [1.5, 1.4, 1.3, 1.2, 1.1, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.2, 0.1, 0.01, 0]
    # for sillhuette to avoid 1 cluster results
    silhouette_d_s = [1.4, 1.3, 1.2, 1.1, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]

    optimal_ds_en = [(1, 1), (0.9, 0.7), (0.8, 0.01), (0.9, 0.8), (0.9, 1.2)]
    optimal_ds_all = [1, 0.9, 0.8, 1, 0.9]
    # d_s = optimal_ds
    # d_s = [None]

    maxn = None

    # for maxd in d_s:
    folder_en = 'data/intervals_ifrim/'
    folder_all = 'data/all_multilingual_prep/'
    folder = folder_en
    # writer = csv.writer(open(folder+"results_to_evaluate.csv", 'w'))
    writer = UnicodeWriter(open(folder+"results_to_evaluate.csv", 'w'), quoting=csv.QUOTE_ALL)


    for idx, interval in enumerate(intervals):
        print interval
        embeddings_folder = folder + interval + '/'
        # print d_s[idx]
        # padding between intervals
        # hierarchical_clustering(interval, embeddings_folder, 'cosine', n_clusters=None, max_ds=d_s, ifrim=True)

        # 1st try all maxds to find optimal
        hierarchical_clustering(interval, embeddings_folder, 'euclidean', n_clusters=maxn, max_ds=all_d_s_tweet2vec, infile='tweet2vec_wo_users_wo_urls', writer=None, calc_v_measure=True, show_topics=False)

        # + calculate silhouette
        # hierarchical_clustering(interval, embeddings_folder, 'euclidean', n_clusters=maxn, max_ds=silhouette_d_s, infile='tweet2vec_wo_users_wo_urls', writer=None, calc_v_measure=False, show_topics=False, silhouette=True)
        # + calculate v-measure in place of silh to compare
        # hierarchical_clustering(interval, embeddings_folder, 'euclidean', n_clusters=maxn, max_ds=silhouette_d_s, infile='tweet2vec_wo_users_wo_urls', writer=None, calc_v_measure=True, show_topics=False, silhouette=False)

        # for certain maxd
        # single
        # TODO run for all intervals vmeasure to check max num clusters and ifrim
        # hierarchical_clustering(interval, embeddings_folder, 'euclidean', n_clusters=maxn, max_ds=[0], infile='tweet2vec_wo_users_wo_urls', writer=writer)
        # hierarchical_clustering(interval, embeddings_folder, 'euclidean', n_clusters=maxn, max_ds=[0], infile='ifrim', writer=writer)

        # 2nd from array both write into csv for manual evaluation
        # print optimal_ds[idx][0]
        # hierarchical_clustering(interval, embeddings_folder, 'euclidean', n_clusters=maxn, max_ds=[optimal_ds_en[idx][0]], infile='tweet2vec_wo_users_wo_urls', writer=writer, v_measure=False, show_topics=True)
        # # print optimal_ds[idx][1]
        # hierarchical_clustering(interval, embeddings_folder, 'euclidean', n_clusters=maxn, max_ds=[optimal_ds_en[idx][1]], infile='ifrim', writer=writer, v_measure=False, show_topics=True)

        # for all lang dataset
        # hierarchical_clustering(interval, embeddings_folder, 'euclidean', n_clusters=maxn, max_ds=[optimal_ds_all[idx]], infile='tweet2vec_wo_users_wo_urls', writer=writer, v_measure=False, show_topics=True)


        ## load_embeddings(embeddings)

if __name__ == '__main__':
    folder_en = 'data/intervals_ifrim/'
    sample_embeddings = folder_en + '01:30/embeddings_tweet2vec_wo_users_wo_urls2.npy'
        # # embeddings='../../embeddings_18snow_wourls'
        # # embeddings_folder='data/18:00/'
    # PCA(embeddings)
    vectors = reduce_dimensionality_tsne(sample_embeddings)
    # y = np.hstack([digits.target[digits.target==i]
               # for i in range(10)])
    scatter(vectors)
    plt.savefig('tsne-ifrim-01:30-tweet2vec.png', dpi=120)
