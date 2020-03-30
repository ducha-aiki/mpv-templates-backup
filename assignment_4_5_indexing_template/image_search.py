from scipy.sparse import csr_matrix
import numpy as np


def kmeans(num_clusters: int, data: np.ndarray, max_num_iterations: int = 1000) -> (np.ndarray, float):
    """
    Cluster data [num_points, num_dims] into num_clusters, using K-Means algorithm.
    If not converged yet, stop after max_num_iterations.

    clusters: [num_clusters, num_dims]
    :param num_clusters:
    :param data:
    :param max_num_iterations:
    :return: (clusters, distances_sum)
    """
    num_dims = data.shape[1]
    clusters = np.zeros((num_clusters, num_dims))
    distances_sum = -1.0


    return clusters, distances_sum


def nearest(means: np.ndarray, data: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    For each data-point in data [num_points, num_dims] find nearest mean [num_means, num_dims].
    Return indices [num_points, ] and distances [num_points, ] to the nearest mean for each data-point.
    :param means:
    :param data:
    :return: (idxs, distances)
    """
    idxs = np.zeros((data.shape[0], ), dtype=np.uint)
    distances = np.zeros((data.shape[0], ))

    return idxs, distances


def create_db(imgs_visual_words: np.ndarray, num_words: int) -> csr_matrix:
    """
    Create database [num_words, num_imgs] of word weights represented as csr_matrix. Details explained at tutorial page.
    imgs_visual_words is of dimensions [num_imgs, visual_words_in_img]. Number of visual_words_in_img differs for
    each img.

    :param imgs_visual_words:
    :param num_words:
    :return: db
    """
    num_imgs = imgs_visual_words.shape[0]
    db = csr_matrix(np.zeros((num_words, num_imgs)))

    return db


def create_db_tfidf(imgs_visual_words: np.ndarray, num_words: int, idf: np.ndarray) -> csr_matrix:
    """
    Create database [num_words, num_imgs] of word weights represented as csr_matrix. Details explained at tutorial page.
    imgs_visual_words is of dimensions [num_imgs, visual_words_in_img]. Number of visual_words_in_img differs for
    each img.
    idf - Inverse Document Frequency. Shape: [num_words, 1]
    :param imgs_visual_words:
    :param num_words:
    :param idf:
    :return: df
    """
    num_imgs = imgs_visual_words.shape[0]
    db = csr_matrix(np.zeros((num_words, num_imgs)))

    return db


def get_idf(imgs_visual_words: np.ndarray, num_words: int) -> np.ndarray:
    """
    Create Inverse Document Frequency of shape: [num_words, num_imgs]
    imgs_visual_words is of dimensions [num_imgs, visual_words_in_img]. Number of visual_words_in_img differs for
    each img.
    idf is of shape: [num_words, 1]

    :param imgs_visual_words:
    :param num_words:
    :return: idf
    """
    idf = np.zeros((num_words, 1))

    return idf


def query(db: csr_matrix, query: np.ndarray, idf: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Sort imgs in descending order by similarity with a query [num_words_in_query_img, ].
    Return img indices [num_imgs, ] and scores [num_imgs, ] ordered in descending order sorted by scores.
    :param db:
    :param query:
    :param idf:
    :return: (idxs, scores)
    """

    idxs = np.zeros((db.shape[1], ), dtype=np.uint)
    scores = np.zeros((db.shape[1], ))

    return idxs, scores
