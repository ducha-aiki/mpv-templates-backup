import numpy as np, os, cv2, PIL.Image, random
import matplotlib.pyplot as plt


def get_query_data(visual_words, geometries, q_id, bbox):
    query_geometry = geometries[q_id]
    query_visual_words = visual_words[q_id]
    qidxs_inbox = get_pts_in_box(query_geometry, bbox)  # choose relevant part from an query image
    query_visual_words_inbox = query_visual_words[qidxs_inbox]
    query_geometry_inbox = query_geometry[qidxs_inbox.reshape(-1)]

    return query_visual_words_inbox, query_geometry_inbox


def get_shortlist_data(visual_words, geometries, candidate_ids):
    return [visual_words[x] for x in candidate_ids], [geometries[x] for x in candidate_ids]


def get_A_matrix_from_geom(geom):
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


def get_pts_in_box(geom, bbox_xyxy):
    """
    geom: of shape [Nx6], where each row corresponds to [x, y, a11, a12, a21, a22]
    bbox_xyxy: bounding box definition in format [x, y, xx, yy]
    return ->
    indices of points inside given bounding box
    """

    x, y, xx, yy = bbox_xyxy

    idxs = np.logical_and(
        np.logical_and(geom[:, 0] >= x, geom[:, 0] <= xx),
        np.logical_and(geom[:, 1] >= y, geom[:, 1] <= yy)
    )

    return np.argwhere(idxs)


def draw_bbox(img, bbox, A=np.eye(3), offset=(0, 0)):
    corner_pts = np.array([
        [bbox[0], bbox[1], 1],
        [bbox[2], bbox[1], 1],
        [bbox[2], bbox[3], 1],
        [bbox[0], bbox[3], 1],
        [bbox[0], bbox[1], 1],
    ], dtype=np.int32)

    corner_pts = np.dot(A, corner_pts.T).T

    corner_pts[:, 0] += offset[0]
    corner_pts[:, 1] += offset[1]

    x, y = corner_pts[0, :2].astype(np.int32)
    img = cv2.circle(img, (x, y), 10, (20, 20, 20), -1)

    for i in range(4):
        pt1 = corner_pts[i, :2].astype(np.int32)
        pt2 = corner_pts[i + 1, :2].astype(np.int32)
        img = cv2.line(img, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (235, 146, 32 + i * 70), 5)

    return img


def draw_tentative_correspondencies(img1, q_geom, img2, d_geom, correspondencies, inliers=None,
                                    draw_outliers_line=True):
    im_visu = np.hstack((img1, img2))
    gray = cv2.cvtColor(im_visu, cv2.COLOR_RGB2GRAY)
    im_visu = np.dstack((gray, gray, gray))

    if inliers is None:
        inliers = np.zeros((correspondencies.shape[0],))

    for (q, d), is_inlier in zip(correspondencies, inliers):
        x, y = int(q_geom[q, 0]), int(q_geom[q, 1])
        x2, y2 = int(d_geom[d, 0]), int(d_geom[d, 1])
        if draw_outliers_line or is_inlier:
            im_visu = cv2.line(
                im_visu, (x, y), (x2 + img1.size[0], y2),
                (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            )

        c = (255, 0, 0)
        if is_inlier:
            c = (0, 255, 0)
        im_visu = cv2.circle(im_visu, (x, y), 3, c, -1)
        im_visu = cv2.circle(im_visu, (x2 + img1.size[0], y2), 3, c, -1)

    return im_visu


def vis_results(img_names, q_id, bbox_xyxy, img_ids, scores, transformations, figure_filename='fig.png'):
    query_img = PIL.Image.open(img_names[q_id])
    visu1 = draw_bbox((np.array(query_img)).astype(np.uint8), bbox_xyxy)
    plt.figure(figsize=(7, 7))
    plt.subplot(3, 2, 1)
    plt.imshow(visu1)
    plt.title("query image, id: {}".format(q_id))
    plt.axis('off')

    for i, (img_id, inl, A) in enumerate(zip(img_ids[:5], scores[:5], transformations[:5])):
        plt.subplot(3, 2, i + 2)
        img = PIL.Image.open(img_names[img_id])
        visu = draw_bbox((np.array(img)).astype(np.uint8), bbox_xyxy, A)
        plt.title('img_id: {}, #inliers: {}'.format(img_id, inl))
        plt.imshow(visu)
        plt.axis('off')
    plt.show()
    plt.savefig(figure_filename, dpi=300)
