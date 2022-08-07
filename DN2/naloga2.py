from multiprocessing.spawn import prepare
from os import listdir
from os.path import join
from re import S
import numpy as np
import math
from torch import embedding
from unidecode import unidecode
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE
import time


def terke(text, n):
    t_dict = {}
    text = unidecode(text.lower())
    for i in range(len(text)-n+1):
        n_terka = text[i:i+n]
        if n_terka not in t_dict:
            t_dict[n_terka] = 1
        else:
            t_dict[n_terka] += 1

    t_dict = {k: v for k, v in sorted(t_dict.items(), key=lambda item: item[1], reverse=True)}
    return t_dict

def read_data(n_terke):
    # Prosim, ne spreminjajte te funkcije. Vso potrebno obdelavo naredite
    # v funkciji terke.
    lds = {}
    for fn in listdir("jeziki2"):
        if fn.lower().endswith(".txt"):
            with open(join("jeziki2", fn), encoding="utf8") as f:
                text = f.read()
                nter = terke(text, n=n_terke)
                lds[fn] = nter
    return lds

def cosine_dist(d1, d2):
    val1 = []
    val2 = []

    common_keys = d1.keys() & d2.keys()

    cvals = [(d1[x], d2[x]) for x in common_keys]
    val1, val2 = map(list, zip(*cvals))

    if len(val1) == 0:
        return 1
    else:
        return (1 - np.dot(val1, val2)/(np.linalg.norm(val1)*np.linalg.norm(val2)))

def prepare_data_matrix(data_dict):
    idf_dict = idf(data_dict)

    data_matrix = np.zeros((len(data_dict), 100))
    counterX = 0
    counterY = 0

    languages = list(data_dict.keys())
    for dokument in data_dict.values():
        for terka in idf_dict:
            try:
                data_matrix[counterX, counterY] = dokument[terka[0]]
            except:
                data_matrix[counterX, counterY] = 0
            counterY+=1
        counterY = 0
        counterX += 1
	
    return data_matrix, languages

def normalize(x):
    val = abs(x).max()
    vec = x / x.max()
    return val, vec

def power_iteration(X):
    """
    Compute the eigenvector with the greatest eigenvalue
    of the covariance matrix of X (a numpy array).

    Return two values:
    - the eigenvector (1D numpy array) and
    - the corresponding eigenvalue (a float)
    """

    X = np.cov(X.T)
    mean = np.mean(X, axis=1)
    for _ in range(50):
        mean = np.dot(X, mean)
        val, mean = normalize(mean)
            
    mean = mean/np.linalg.norm(mean)
    return mean, val

def power_iteration_two_components(X):
    """
    Compute first two eigenvectors and eigenvalues with the power iteration method.
    This function should use the power_iteration function internally.

    Return two values:
    - the two eigenvectors (2D numpy array, each eigenvector in a row) and
    - the corresponding eigenvalues (a 1D numpy array)
    """
    first_vec, first_val = power_iteration(X.copy())
    
    fvec = first_vec.reshape(1, -1)
    X = X - X @ fvec.T * fvec
    second_vec, second_val = power_iteration(X)

    vectors = []
    vectors.append(first_vec)
    vectors.append(second_vec)
    
    values = []
    values.append(first_val)
    values.append(second_val)
    return vectors, values

def project_to_eigenvectors(X, vecs):
    """
    Project matrix X onto the space defined by eigenvectors.
    The output array should have as many rows as X and as many columns as there
    are vectors.
    """
    mean = np.mean(X, axis=0)
    X_pca = np.empty((X.shape[0], len(vecs)))
    for i in range(X.shape[0]):
        X_pca[i] = vecs @ (X[i] - mean)

    return X_pca

def total_variance(X):
    """
    Total variance of the data matrix X. You will need to use for
    to compute the explained variance ratio.
    """
    return np.var(X, axis=0, ddof=1).sum()

def explained_variance_ratio(X, eigenvectors, eigenvalues):
    return (np.sum(eigenvalues)/total_variance(X))

def dissimilarity_matrix(data):
    n = len(data)
    diss_mat = np.zeros((n, n))
    val = list(data.values())

    for i in range(n):
        for j in range(i+1, n, 1):
            if(diss_mat[i, j] != 0 or diss_mat[i, j] != 0):
                continue

            cos = cosine_dist(val[i], val[j])
            diss_mat[i, j] = cos
            diss_mat[j, i] = cos 

    return diss_mat

def plot_PCA():
    """
    Everything (opening files, computation, plotting) needed
    to produce a plot of PCA on languages data.
    """
    data = read_data(3)
    data_matrix, languages = prepare_data_matrix(data)
    eigenvec, eigenvalue = power_iteration_two_components(data_matrix)

    projection = project_to_eigenvectors(data_matrix, eigenvec)
    explained_var = explained_variance_ratio(data_matrix, eigenvec, eigenvalue)

    x = []
    y = []

    for el in projection:
        x.append(el[0])
        y.append(el[1])
    
    plt.scatter(x,y)

    for i in range(len(x)):
        plt.text(x[i], y[i], languages[i][:2])

    plt.title(explained_var)
    plt.show()

def plot_MDS():
    """
    Everything (opening files, computation, plotting) needed
    to produce a plot of MDS on languages data.

    Use sklearn.manifold.MDS and explicitly run it with a distance
    matrix obtained with cosine distance on full triplets.
    """
    data = read_data(3)
    languages = list(data.keys())
    diss_mat = dissimilarity_matrix(data)

    mds = MDS(n_components=2,verbose=1,eps=1e-5, dissimilarity='precomputed', random_state=1)
    mds.fit(diss_mat)

    plt.scatter(mds.embedding_[:,0], mds.embedding_[:,1]) 

    for i in range(len(diss_mat)):
        plt.text(mds.embedding_[:,0][i], mds.embedding_[:,1][i], languages[i][:2])

    plt.show()

def plot_TSNE():
    data = read_data(3)
    languages = list(data.keys())
    start_time = time.time()
    diss_mat = dissimilarity_matrix(data)
    print(time.time() - start_time)


    tsne = TSNE(n_components=2, verbose=1)
    tsne.fit(diss_mat)

    plt.scatter(tsne.embedding_[:,0], tsne.embedding_[:,1]) 

    for i in range(len(diss_mat)):
        plt.text(tsne.embedding_[:,0][i], tsne.embedding_[:,1][i], languages[i][:2])

    plt.show()

def get_number_of_docs(data, term):
    counter = 0
    for docs in data.keys():
        if term in data[docs]:
            counter += 1 
    return counter

def idf(data):
    idf_dict = {}
    N = len(data)

    for docs in data.keys():
        for term in data[docs]:

            if term in idf_dict:
                continue
            
            num_of_docs_with_term = get_number_of_docs(data, term)
            idf_value = 0

            if num_of_docs_with_term != 0:
                idf_value = math.log(N/num_of_docs_with_term)

            idf_dict[term] = idf_value
    
    idf_dict = sorted(idf_dict.items(), key=lambda item: item[1], reverse=False)[:100]
    return idf_dict

if __name__ == "__main__":
    plot_MDS()