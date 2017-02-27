from gensim.models import Word2Vec, KeyedVectors
import numpy as np
from sklearn.cluster import KMeans

#Load word2vec
W2V_PATH = "E:\\Datascience\\Word2VecWeights\\GoogleNews-vectors-negative300.bin"
model = KeyedVectors.load_word2vec_format(W2V_PATH, binary=True)

def extract_main_word(line):
    words = line[:-1]
    first_word = words.split(",")[0]
    first_word = first_word.replace(" ", "_")
    return first_word

#Get word embeddings
with open("core6K.txt", "r") as f:
    #Discard newline chars and only keep first word of each line
    words = [extract_main_word(line) for line in f][:2000]

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

print(ignored_num)


embedded_words = np.array(embedded_words)
embeddings = np.vstack(individual_embeddings)
np.save("words.npy", embedded_words)
np.save("embeddings.npy", embeddings)
# embeddings = np.load("embeddings.npy")
# embedded_words = np.load("words.npy")

#Cluster with k-means
N_CLUSTER = 30
KM = KMeans(n_clusters=N_CLUSTER, n_init=100)
clusters = KM.fit_predict(embeddings)


cluster_text = ""
for i in range(N_CLUSTER):
    cluster_text += str(i) + "\n"
    for j in range(clusters.shape[0]):
        cluster = clusters[j]
        word = embedded_words[j]
        if cluster == i:
            cluster_text += word + "\n"

with open("clusters2.txt", "w") as f:
    f.write(cluster_text)
