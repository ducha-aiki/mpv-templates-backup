import cv2
import time, os
import numpy as np
from spatial_verification import *
import PIL.Image
import matplotlib.pyplot as plt


def get_pts_in_box(geom, bbox_xyxy):
    """
    :param geom: of shape [Nx6], where each row corresponds to [x, y, a11, a12, a21, a22]
    :param bbox_xyxy: bounding box definition in format [x, y, xx, yy]
    :return: indexes of points inside given bounding box
    """

    x, y, xx, yy = bbox_xyxy

    idxs = np.logical_and(
        np.logical_and(geom[:, 0] >= x, geom[:, 0] <= xx),
        np.logical_and(geom[:, 1] >= y, geom[:, 1] <= yy)
    )

    return np.argwhere(idxs)


def draw_tentative_correspondencies(img1, q_geom, img2, d_geom, correspondencies, inliers=None, draw_outliers_line=True):
    import random
    import cv2

    im_visu = np.hstack((img1, img2))
    gray = cv2.cvtColor(im_visu, cv2.COLOR_RGB2GRAY)
    im_visu = np.dstack((gray, gray, gray))

    if inliers is None:
        inliers = np.zeros((correspondencies.shape[0], ))

    for (q, d), is_inlier in zip(correspondencies, inliers):
        x, y = int(q_geom[q, 0]), int(q_geom[q, 1])
        x2, y2 = int(d_geom[d, 0]), int(d_geom[d, 1])
        if draw_outliers_line or is_inlier:
            im_visu = cv2.line(
                im_visu, (x, y), (x2 + img1.size[0],  y2),
                (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            )

        c = (255, 0, 0)
        if is_inlier:
            c = (0, 255, 0)
        im_visu = cv2.circle(im_visu, (x, y), 3, c, -1)
        im_visu = cv2.circle(im_visu, (x2 + img1.size[0], y2), 3, c, -1)

    return im_visu


def draw_bbox(img, bbox, A=np.eye(3), offset=(0, 0)):
    corner_pts = np.array([
        [bbox[0], bbox[1], 1],
        [bbox[2], bbox[1], 1],
        [bbox[2], bbox[3], 1],
        [bbox[0], bbox[3], 1],
        [bbox[0], bbox[1], 1],
    ], dtype=np.int)

    corner_pts = np.dot(A, corner_pts.T).T

    corner_pts[:, 0] += offset[0]
    corner_pts[:, 1] += offset[1]

    x, y = corner_pts[0, :2].astype(np.int)
    img = cv2.circle(img, (x, y), 10, (20, 20, 20), -1)

    for i in range(4):
        pt1 = corner_pts[i, :2].astype(np.int)
        pt2 = corner_pts[i+1, :2].astype(np.int)
        img = cv2.line(img, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (235, 146, 32 + i*70), 5)

    return img


def get_bbox(img):
    import cv2
    points = []

    def click_positions(event, x, y, flags, param):
        global points
        if event == cv2.EVENT_LBUTTONDOWN:
            print('{{x: {}, y: {}}}'.format(x, y))
            points.append([x, y])

    cv2.namedWindow('im')
    cv2.setMouseCallback('im', click_positions)
    visu = img.copy()

    while True:
        cv2.imshow("im", img)
        k = cv2.waitKey(0)

        if k == 27:
            break

    print(points)
    return points


def query_and_visu(q_id, visual_words, geometries, bbox_xyxy, DB, idf, options, img_names, figsize=(7, 7)):
    t = time.time()
    scores, img_ids, As = query_spatial_verification(
        visual_words[q_id], geometries[q_id], bbox_xyxy, visual_words, geometries, DB, idf, options
    )
    print("computed in: {:.3f}s".format(time.time() - t))

    query_img = PIL.Image.open(os.path.join(options['data_root_dir'], img_names[q_id]))
    visu1 = draw_bbox((np.array(query_img)).astype(np.uint8), bbox_xyxy)
    plt.figure(figsize=figsize)
    plt.subplot(3, 2, 1)
    plt.imshow(visu1)
    plt.title("query image, id: {}".format(q_id))
    plt.axis('off')

    for i, (img_id, inl, A) in enumerate(zip(img_ids[:5], scores[:5], As[:5])):
        plt.subplot(3, 2, i+2)
        img = PIL.Image.open(os.path.join(options['data_root_dir'], img_names[img_id]))
        visu = draw_bbox((np.array(img)).astype(np.uint8), bbox_xyxy, A)
        plt.title('img_id: {}, #inliers: {}'.format(img_id, inl))
        plt.imshow(visu)
        plt.axis('off')

    plt.show()


if __name__ == '__main__':
    bbox_xyxy = [780, 625, 940, 720]
    img = draw_bbox(np.zeros((1080, 1920, 3), dtype=np.uint8), bbox_xyxy, np.array([[0.5, 0, 0], [0, 0.3, 0], [0, 0, 1]]))
    import matplotlib.pyplot as plt

    plt.imshow(img)
    plt.show()