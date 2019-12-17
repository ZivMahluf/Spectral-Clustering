import numpy as np
import matplotlib.pyplot as plt
import pickle


def circles_example():
    """
    an example function for generating and plotting synthetic data.
    """

    t = np.arange(0, 2 * np.pi, 0.03)
    length = np.shape(t)
    length = length[0]
    circle1 = np.matrix([np.cos(t) + 0.1 * np.random.randn(length),
                         np.sin(t) + 0.1 * np.random.randn(length)])
    circle2 = np.matrix([2 * np.cos(t) + 0.1 * np.random.randn(length),
                         2 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle3 = np.matrix([3 * np.cos(t) + 0.1 * np.random.randn(length),
                         3 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle4 = np.matrix([4 * np.cos(t) + 0.1 * np.random.randn(length),
                         4 * np.sin(t) + 0.1 * np.random.randn(length)])
    circles = np.concatenate((circle1, circle2, circle3, circle4), axis=1)

    return circles


def apml_pic_example(path='APML_pic.pickle'):
    """
    an example function for loading and plotting the APML_pic data.

    :param path: the path to the APML_pic.pickle file
    """

    with open(path, 'rb') as f:
        apml = pickle.load(f)

    plt.plot(apml[:, 0], apml[:, 1], '.')
    plt.show()


def microarray_exploration(data_path='microarray_data.pickle',
                           genes_path='microarray_genes.pickle',
                           conds_path='microarray_conds.pickle'):
    """
    an example function for loading and plotting the microarray data.
    Specific explanations of the data set are written in comments inside the
    function.

    :param data_path: path to the data file.
    :param genes_path: path to the genes file.
    :param conds_path: path to the conds file.
    """

    # loading the data:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open(genes_path, 'rb') as f:
        genes = pickle.load(f)
    with open(conds_path, 'rb') as f:
        conds = pickle.load(f)

    return data


def euclid(X, Y):
    """
    return the pair-wise euclidean distance between two data matrices.
    :param X: NxD matrix.
    :param Y: MxD matrix.
    :return: NxM euclidean distance matrix.
    """
    result = np.ndarray(shape=(len(X), len(Y)))
    for idx, x in enumerate(X):
        result[idx] = np.linalg.norm(Y - x, axis=-1)
    return result


def euclidean_centroid(X):
    """
    return the center of mass of data points of X.
    :param X: a sub-matrix of the NxD data matrix that defines a cluster.
    :return: the centroid of the cluster.
    """
    return X.mean(axis=0)


def kmeans_pp_init(X, k, metric):
    """
    The initialization function of kmeans++, returning k centroids.
    :param X: The data matrix.
    :param k: The number of clusters.
    :param metric: a metric function like specified in the kmeans documentation.
    :return: kxD matrix with rows containing the centroids.
    """
    # pick first centroid at random
    centroids = np.ndarray(shape=(k, len(X[0])))
    centroids[0] = X[np.random.randint(len(X), size=1)]
    for i in range(1, k):
        # calculate dist matrix for all remaining
        min_distances = np.min(metric(X, centroids), axis=1)
        normalized_rand_probs = (min_distances ** 2) / np.sum(min_distances ** 2)
        normalized_rand_probs[np.isnan(normalized_rand_probs)] = 0
        centroids[i] = X[np.random.choice(len(X), size=1, p=normalized_rand_probs)]
    return centroids


def kmeans(X, k, iterations=10, metric=euclid, center=euclidean_centroid, init=kmeans_pp_init):
    """
    The K-Means function, clustering the data X into k clusters.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param iterations: The number of iterations.
    :param metric: A function that accepts two data matrices and returns their
            pair-wise distance. For a NxD and KxD matrices for instance, return
            a NxK distance matrix.
    :param center: A function that accepts a sub-matrix of X where the rows are
            points in a cluster, and returns the cluster centroid.
    :param init: A function that accepts a data matrix and k, and returns k initial centroids.
    :return: a tuple of (clustering, centroids, statistics)
    clustering - A N-dimensional vector with indices from 0 to k-1, defining the clusters.
    centroids - The kxD centroid matrix.
    """
    best_loss, best_centroids, best_clustering = np.inf, None, None

    for i in range(iterations):
        centroids = init(X, k, metric)
        last_clustering = np.zeros(shape=len(X))
        while True:
            clustering = np.argmin(euclid(X, centroids), axis=1)
            if np.equal(clustering, last_clustering).all():  # didn't change
                break
            for cluster in range(k):  # update
                if X[clustering == cluster].any():  # if set is empty - don't update
                    centroids[cluster] = center(X[clustering == cluster])
            last_clustering = clustering
        # converged -- no change was made in last step
        loss = elbow(X, k, centroids, clustering)
        if loss < best_loss:
            best_loss, best_centroids, best_clustering = loss, centroids, clustering

    return best_clustering, best_centroids, None


def gaussian_kernel(X, sigma):
    """
    calculate the gaussian kernel similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param sigma: The width of the Gaussian in the heat kernel.
    :return: NxN similarity matrix.
    """
    return np.power(np.e, - (X ** 2) / (2 * (sigma ** 2)))


def mnn(X, m):
    """
    calculate the m nearest neighbors similarity of the given distance matrix.
    :param X: A NxN distance matrix.
    :param m: The number of nearest neighbors.
    :return: NxN similarity matrix.
    """
    similarity_matrix = np.zeros_like(X)
    # find m nearest neighbors for each sample
    nearest_neighbors = np.argpartition(X, m, axis=1)
    idx = 0
    for neighbors in nearest_neighbors:
        similarity_matrix[idx, neighbors[:m]] = 1
        similarity_matrix[neighbors[:m], idx] = 1
        idx += 1
    return similarity_matrix


def spectral(X, k, similarity_param, similarity=gaussian_kernel):
    """
    Cluster the data into k clusters using the spectral clustering algorithm.
    :param X: A NxD data matrix.
    :param k: The number of desired clusters.
    :param similarity_param: m for mnn, sigma for the Gaussian kernel.
    :param similarity: The similarity transformation of the data.
    :return: clustering, as in the kmeans implementation.
    """
    # calculate dist matrix
    dist_matrix = euclid(X, X)
    # get similarity matrix
    similarity_matrix = similarity(dist_matrix, similarity_param)
    # get diagonal degree matrix
    diagonal_matrix = np.diag(np.sum(similarity_matrix, axis=1))
    # laplacian matrix by given formula
    inv_diagonal = np.sqrt(np.linalg.pinv(diagonal_matrix))
    laplacian_matrix = np.identity(len(X)) - inv_diagonal @ similarity_matrix @ inv_diagonal
    # get lowest eigen vectors
    w, v = np.linalg.eigh(laplacian_matrix)
    eigen_vectors = v[:, np.argpartition(w, k)[:k]]
    return eigen_vectors


# -------------- Added functions --------------

def choose_sigma(dist_matrix):
    # get and plot histogram
    hist, bin_edges = np.histogram(dist_matrix)
    plt.hist(hist)
    plt.title("histogram of sigma values")
    plt.show()
    plt.cla()
    return bin_edges


def silhouette(X, k, clustering, dist_matrix):
    """
    return the silhouette score
    :param k: amount of clusters
    :param clustering: the clustering (size samples)
    :param dist_matrix: the distance matrix, to avoid recalculating
    :return: the silhouette score
    """
    silhouette_scores = np.zeros(shape=len(X))
    for sample_index, sample in enumerate(X):
        sample_cluster_index = clustering[sample_index]
        samples_indices_in_same_cluster = [x[0] for x in np.argwhere(clustering == sample_cluster_index)]
        # calculate a
        if len(samples_indices_in_same_cluster) == 1:
            a = 0  # distance from itself is 0
        else:
            a = np.sum(dist_matrix[sample_index][samples_indices_in_same_cluster]) / (
                    len(samples_indices_in_same_cluster) - 1)
        # find distance from all other clusters
        min_b = np.inf
        b = 0
        for c in range(k):
            if c != sample_cluster_index:  # for all other clusters
                indices_of_samples_in_other_cluster = [x[0] for x in np.argwhere(clustering == c)]
                b = np.mean(dist_matrix[sample_index][indices_of_samples_in_other_cluster])
                if b < min_b:
                    min_b = b
        silhouette_scores[sample_index] = (b - a) / np.fmax(a, b)
    return np.mean(silhouette_scores)


def eigen_gap(eigenvalues):
    """
    return the index where the largest eigen value gap occurs
    :param eigenvalues: the eigenvalues
    :return: approximate k
    """
    return np.argmax(np.diff(eigenvalues))


def get_synth_data():
    x1 = np.random.normal(loc=1, scale=1, size=50)
    y1 = np.random.normal(loc=1, scale=1, size=50)
    x2 = np.random.normal(loc=9, scale=1, size=50)
    y2 = np.random.normal(loc=9, scale=1, size=50)
    x3 = np.random.normal(loc=0, scale=1, size=50)
    y3 = np.random.normal(loc=8, scale=1, size=50)

    # plt.scatter(x1, y1, color='b')
    # plt.scatter(x2, y2, color='r')
    # plt.scatter(x3, y3, color='g')
    # plt.show()

    first = [(x, y) for x, y in zip(x1, y1)]
    second = [(x, y) for x, y in zip(x2, y2)]
    third = [(x, y) for x, y in zip(x3, y3)]
    return first + second + third


def elbow(X, k, centroids, clustering):
    loss = 0.0
    for i in range(k):
        cluster_indices = [x[0] for x in np.argwhere(clustering == i)]
        samples = X[cluster_indices]
        loss += np.fabs(np.sum(np.sum(samples - centroids[i], axis=0))) / len(cluster_indices)
    return loss


def plot_results(X, clustering, centroids, K, title=""):
    colors = ['red', 'yellow', 'purple', 'green']
    # plot points by clustering
    for k in range(K):
        indices = [x[0] for x in np.argwhere(clustering == k)]
        plt.scatter(X[indices][:, 0], X[indices][:, 1], color=colors[k])
    # plot centroids
    plt.scatter([t[0] for t in centroids], [t[1] for t in centroids], color='black')
    plt.title(title)
    plt.show()
    plt.cla()


def k_means_synth():
    points = get_synth_data()
    K = 3
    X = np.array(points)
    clustering, centroids, _ = kmeans(X, K, iterations=10)
    plot_results(X, clustering, centroids, K, "k means on synth data")


def k_means_circles():
    points = circles_example().T
    K = 4
    X = np.array(points)
    clustering, centroids, _ = kmeans(X, K, iterations=10)
    plot_results(X, clustering, centroids, K, "k means on circles data")


def spectral_synth():
    points = get_synth_data()
    K = 3
    X = np.array(points)
    bin_edges = choose_sigma(euclid(X, X))
    # get different results for different percentiles
    normalized_eigen_vectors = spectral(X, K, similarity_param=bin_edges[7], similarity=gaussian_kernel)
    clustering, centroids, _ = kmeans(normalized_eigen_vectors, K, iterations=10)
    plot_results(X, clustering, centroids, K,
                 "spectral on synth data with gaussian kernel and sigma=%.2f" % bin_edges[7])

    normalized_eigen_vectors = spectral(X, K, similarity_param=bin_edges[2], similarity=gaussian_kernel)
    clustering, centroids, _ = kmeans(normalized_eigen_vectors, K, iterations=10)
    plot_results(X, clustering, centroids, K,
                 "spectral on synth data with gaussian kernel and sigma=%.2f" % bin_edges[2])

    normalized_eigen_vectors = spectral(X, K, similarity_param=bin_edges[4], similarity=gaussian_kernel)
    clustering, centroids, _ = kmeans(normalized_eigen_vectors, K, iterations=10)
    plot_results(X, clustering, centroids, K,
                 "spectral on synth data with gaussian kernel and sigma=%.2f" % bin_edges[4])


def spectral_circles():
    points = circles_example().T
    K = 4
    X = np.array(points)
    bin_edges = choose_sigma(euclid(X, X))
    # get different results for different percentiles
    normalized_eigen_vectors = spectral(X, K, similarity_param=bin_edges[7], similarity=gaussian_kernel)
    clustering, centroids, _ = kmeans(normalized_eigen_vectors, K, iterations=10)
    plot_results(X, clustering, centroids, K,
                 "spectral on synth data with gaussian kernel and sigma=%.2f" % bin_edges[7])

    normalized_eigen_vectors = spectral(X, K, similarity_param=0.1, similarity=gaussian_kernel)
    clustering, centroids, _ = kmeans(normalized_eigen_vectors, K, iterations=10)
    plot_results(X, clustering, centroids, K,
                 "spectral on synth data with gaussian kernel and sigma=%.2f" % 0.1)

    normalized_eigen_vectors = spectral(X, K, similarity_param=bin_edges[4], similarity=gaussian_kernel)
    clustering, centroids, _ = kmeans(normalized_eigen_vectors, K, iterations=10)
    plot_results(X, clustering, centroids, K,
                 "spectral on synth data with gaussian kernel and sigma=%.2f" % bin_edges[4])


def plotting_similarity_matrix():
    points = get_synth_data()  # this is the ordered, clustered data
    K = 3
    X = np.array(points)
    clustered_w = mnn(euclid(X, X), 2)

    # shuffle X
    np.random.shuffle(X)
    shuffled_w = mnn(euclid(X, X), 2)

    plt.matshow(shuffled_w)
    plt.show()
    plt.cla()
    plt.matshow(clustered_w)
    plt.show()
    plt.cla()


def choose_k():
    # get k silhouette values for different k's and plot their values
    points = get_synth_data()
    X = np.array(points)
    silhouette_scores = []
    dist_matrix = euclid(X, X)
    for k in range(1, 10):
        clustering, centroids, _ = kmeans(X, k, iterations=10)
        silhouette_scores.append(silhouette(X, k, clustering, dist_matrix))
    # plot the scores
    plt.plot(range(1, 10), silhouette_scores)
    plt.title("silhouette scores by k")
    plt.show()


def biological_clustering():
    data = microarray_exploration()

    silhouette_kmeans_scores = np.zeros(shape=15)
    silhouette_spectral_gaussian_scores = np.zeros(shape=15)
    silhouette_spectral_mnn_scores = np.zeros(shape=15)
    elbows_kmeans_scores = np.zeros(shape=15)
    elbows_spectral_gaussian_scores = np.zeros(shape=15)
    elbows_spectral_mnn_scores = np.zeros(shape=15)

    dist_matrix = euclid(data, data)
    sigma_param = choose_sigma(dist_matrix)[7]
    mnn_param = 5

    for k in range(1, 16):
        # calculate for k means
        clustering, centroids, _ = kmeans(data, k, iterations=1)
        silhouette_kmeans_scores[k - 1] = silhouette(data, k, clustering, dist_matrix)
        elbows_kmeans_scores[k - 1] = elbow(data, k, centroids, clustering)
        print(silhouette_kmeans_scores)
        print(elbows_kmeans_scores)

        # calculate for spectral with gaussian
        normalized_eigen_vectors = spectral(data, k, similarity_param=sigma_param, similarity=gaussian_kernel)
        clustering, centroids, _ = kmeans(normalized_eigen_vectors, k, iterations=1)
        silhouette_spectral_gaussian_scores[k - 1] = silhouette(data, k, clustering, dist_matrix)
        elbows_spectral_gaussian_scores[k - 1] = elbow(data, k, centroids, clustering)

        # calculate for spectral with mnn
        normalized_eigen_vectors = spectral(data, k, similarity_param=mnn_param, similarity=mnn)
        clustering, centroids, _ = kmeans(normalized_eigen_vectors, k, iterations=1)
        silhouette_spectral_mnn_scores[k - 1] = silhouette(data, k, clustering, dist_matrix)
        elbows_spectral_mnn_scores[k - 1] = elbow(normalized_eigen_vectors, k, centroids, clustering)

        print("mnn sil:" + str(silhouette_spectral_mnn_scores))
        print("mnn elbow:" + str(elbows_spectral_mnn_scores))
        print("gaus sil:" + str(silhouette_spectral_gaussian_scores))
        print("gaus elbow:" + str(elbows_spectral_gaussian_scores))

    # plot graphs for results
    fig, axs = plt.subplots(1, 2)
    x = range(1, 16)
    axs[0].plot(x, silhouette_kmeans_scores, label='kmeans_silhouette_scores', color='blue')
    axs[1].plot(x, elbows_kmeans_scores, label='kmeans_elbows_scores', color='green')
    fig.suptitle("k means silhouette and elbows scores by k")
    plt.show()
    plt.cla()

    fig, axs = plt.subplots(2, 2)
    x = range(1, 16)
    axs[0][0].plot(x, silhouette_spectral_gaussian_scores, label='spectral gaussian silhouette', color='blue')
    axs[0][1].plot(x, silhouette_spectral_mnn_scores, label='spectral mnn silhouette', color='green')
    axs[1][0].plot(x, elbows_spectral_gaussian_scores, label='spectral gaussian elbow', color='blue')
    axs[1][1].plot(x, elbows_spectral_mnn_scores, label='spectral mnn elbow', color='green')
    fig.suptitle("Spectral clustering with gaussian and mnn")
    plt.show()
    plt.cla()


def tsne():
    from sklearn.datasets import load_digits
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    data, target = load_digits(return_X_y=True)

    # TSNE
    embedded_data = TSNE(n_components=2).fit_transform(data)

    for i in range(10):  # possible digits
        sample_indices = [idx[0] for idx in np.argwhere(target == i)]
        plt.scatter(embedded_data[sample_indices, 0], embedded_data[sample_indices, 1])
    plt.show()
    plt.cla()

    # PCA
    embedded_data = PCA(n_components=2).fit_transform(data)

    for i in range(10):  # possible digits
        sample_indices = [idx[0] for idx in np.argwhere(target == i)]
        plt.scatter(embedded_data[sample_indices, 0], embedded_data[sample_indices, 1])
    plt.show()
    plt.cla()

    # ---------------------- use synthetic data ----------------------
    data = get_synth_data()
    target = np.concatenate(
        [np.full(shape=50, fill_value=0), np.full(shape=50, fill_value=1), np.full(shape=50, fill_value=2)], axis=0)

    # TSNE
    embedded_data = TSNE(n_components=1).fit_transform(data)

    for i in range(3):  # possible digits
        sample_indices = [idx[0] for idx in np.argwhere(target == i)]
        plt.plot(embedded_data[sample_indices])
    plt.ylabel("embedded value of TSNE")
    plt.show()
    plt.cla()

    # PCA
    embedded_data = PCA(n_components=1).fit_transform(data)

    for i in range(3):  # possible digits
        sample_indices = [idx[0] for idx in np.argwhere(target == i)]
        plt.plot(embedded_data[sample_indices])
    plt.ylabel("embedded value of PCA")
    plt.show()
    plt.cla()


if __name__ == '__main__':
    # k_means_synth()
    # k_means_circles()
    # spectral_synth()
    # spectral_circles()
    # plotting_similarity_matrix()
    # choose_k()
    # biological_clustering()
    # tsne()
    pass
