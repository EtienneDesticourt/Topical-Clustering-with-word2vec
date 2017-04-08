# -*- coding: utf-8 -*-
import sys, codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
from sklearn.cluster import KMeans

# Load word2vec
W2V_PATH = "E:\\Datascience\\Word2VecWeights\\GoogleNews-vectors-negative300.bin"
W2V_JAP_PATH = "E:\\Datascience\\Word2VecWeights\\w2v-prettified.txt-{'min_count'_ 10, 'window'_ 10, 'workers'_ 4, 'size'_ 150}.model"
W2V_JAP_PATH_2 = "E:\\Datascience\\Word2VecWeights\\w2v-prettified.txt-{'min_count'_ 10, 'window'_ 7, 'workers'_ 4, 'size'_ 150}.model"
W2V_JAP_PATH_3 = "E:\\Datascience\\Word2VecWeights\\word2vec.gensim.model"
#model = KeyedVectors.load_word2vec_format(W2V_PATH, binary=True)
model = KeyedVectors.load(W2V_JAP_PATH_3).wv

def extract_main_word(line):
    words = line[:-1]
    first_word = words.split(",")[0]
    first_word = first_word.replace(" ", "_")
    return first_word

# Get word embeddings
with open("core6k_jap.txt", "r", encoding="utf-8") as f:
    # Discard newline chars and only keep first word of each line
    words = [extract_main_word(line) for line in f][:2000]

print(len(words), "words")

ignored_num = 0
ignored = []
embedded_words = []
individual_embeddings = []
for word in words:
    try:
        individual_embeddings.append(model[word])
        embedded_words.append(word)
    except KeyError:
        ignored_num += 1
        ignored.append(word)

print(ignored_num, "ignored")


embedded_words = np.array(embedded_words)
embeddings = np.vstack(individual_embeddings)
np.save("words.npy", embedded_words)
np.save("embeddings.npy", embeddings)
# embeddings = np.load("embeddings.npy")
# embedded_words = np.load("words.npy")

# Cluster with k-means
N_CLUSTER = 60
KM = KMeans(n_clusters=N_CLUSTER, n_init=100)

with open("core6K.txt", "r") as f:
    english_words = f.read().split("\n")


mini_bad_clustered = 1000
for k in range(200):

    clusters = KM.fit_predict(embeddings)

    # Transform flat cluster vector into word lists
    word_clusters = {}
    for cluster in range(N_CLUSTER):
        word_clusters[cluster] = []

    for word_index in range(clusters.shape[0]):
        cluster = clusters[word_index]
        word = embedded_words[word_index]
        translation = english_words[words.index(word)]
        word_clusters[cluster].append(word + " : " + translation)

    def calc_num_clusters_by_size(min_size, max_size, clusters):
        num = 0
        for cluster_id in clusters:
            cluster = clusters[cluster_id]
            if len(cluster) > min_size and len(cluster) <= max_size:
                num += 1
        return num

    def calc_num_words_by_cluster_size(min_size, max_size, clusters):
        num = 0
        for cluster_id in clusters:
            cluster = clusters[cluster_id]
            if len(cluster) > min_size and len(cluster) <= max_size:
                num += len(cluster)
        return num

    print("Cluster", str(k), ".")
    print("There are", calc_num_clusters_by_size(0, 20, word_clusters), "clusters of size 0 to 20.")
    print("There are", calc_num_clusters_by_size(20, 40, word_clusters), "clusters of size 20 to 40.")
    print("There are", calc_num_clusters_by_size(40, 60, word_clusters), "clusters of size 40 to 60.")
    print("There are", calc_num_clusters_by_size(60, 80, word_clusters), "clusters of size 60 to 80.")
    print("There are", calc_num_clusters_by_size(80, 10000, word_clusters), "clusters of size > 80.")
    print("There are", calc_num_clusters_by_size(40, 10000, word_clusters), "oversized clusters.")
    print("There are", calc_num_words_by_cluster_size(40, 10000, word_clusters), "badly clustered words.")


    if calc_num_words_by_cluster_size(40, 10000, word_clusters) < mini_bad_clustered:
        mini_bad_clustered = calc_num_words_by_cluster_size(40, 10000, word_clusters)
        # Save to file
        text = str(mini_bad_clustered) + "\n\n"
        for cluster in range(N_CLUSTER):
            text += "\n\n" + str(cluster) + "\n\n"
            text += "\n".join(word_clusters[cluster])

        with open("clusters" + str(k ) + ".txt", "w", encoding="utf-8") as f:
            f.write(text)
