import numpy as np
from scipy.sparse import csr_matrix


def get_tentative_correspondencies(
        query_vw: np.ndarray, vw: np.ndarray, relevant_idxs: np.ndarray, max_tc: int, max_MxN: int) -> np.ndarray:
    """
    Compute tentative correspondencies.

    :param query_vw: [num_keypoints_in_query x 1]
    :param vw: [num_imgs x num_words_in_img] visual_words
    :param relevant_idxs: [num_imgs, ]
    :param max_tc: maximum tentative correspondences
    :param max_MxN: maximum pairs
    :return: correspondencies np.ndarray of shape [num_correspondencies x 2]
    """

    correspondencies = np.zeros((0, 2), dtype=np.int)

    return correspondencies


def get_A_matrix_from_geom(geom: np.ndarray):
    """
    returns a matrix
    [
        a11 a12 x
        a21 a22 y
        0   0   1
    ]

    :param geom:
    :return: 3x3 matrix
    """
    return np.array([
        [geom[2], geom[3], geom[0]],
        [geom[4], geom[5], geom[1]],
        [0, 0, 1],
    ])


def ransac_affine(
        q_geom: np.ndarray, geometries: np.ndarray, correspondencies: np.ndarray, relevant: np.ndarray,
        inlier_threshold: float) -> (np.ndarray, np.ndarray):
    """

    :param q_geom: query geometry of shape [num_keypoints x 6]
    :param geometries: np.ndarray of object of len num_images. each object is np.ndarray [num_keypoints_in_img x 6]
    :param correspondencies: [num_correspondencies x 2]
    :param relevant: relevant indices
    :param inlier_threshold: maximum transformed point to corresponding point distance to be considered as an inlier
    :return: (As, inliers_counts), where As are estimated affine matrices for each image and number of inliers in each image
    """

    K = len(relevant)
    As = np.zeros((K, 3, 3))
    inliers_counts = np.zeros((K, ))

    return As, inliers_counts


def query_spatial_verification(
        query_visual_words: np.ndarray, query_geometry: np.ndarray, bbox: list, visual_words: np.ndarray,
        geometries: np.ndarray, db: csr_matrix, idf: np.ndarray, params: dict) -> (np.ndarray, np.ndarray, np.ndarray):
    """

    :param query_visual_words: [num_keypoints_in_query x 1]
    :param query_geometry: query geometry of shape [num_keypoints x 6]
    :param bbox: list [x, y, xx, yy]
    :param visual_words: [num_imgs x num_words_in_img] visual_words
    :param geometries: np.ndarray of object of len num_images. each object is np.ndarray [num_keypoints_in_img x 6]
    :param db: [num_words, num_imgs]
    :param idf: Inverse Document Frequency. Shape: [num_words, 1]
    :param params: dictionary, important keys here: minimum_score, max_tc, max_MxN, use_query_expansion

    :return:
    """

    num_relevant = ...  # you need to get number based on scores and params['minimum_score']
    inliers_counts = np.zeros((num_relevant, ), dtype=np.int)
    relevant_idxs = np.zeros((num_relevant, ), dtype=np.int)
    As = np.zeros((num_relevant, 3, 3))

    return inliers_counts, relevant_idxs, As

