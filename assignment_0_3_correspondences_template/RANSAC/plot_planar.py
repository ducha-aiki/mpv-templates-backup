from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from dataclasses import dataclass
from matplotlib.patches import Patch
import kornia.feature as KF
from RANSAC.ransac_planar_params import RansacPlanarLog


@dataclass
class PlotParams:
    cross_size_img_size_ratio = 80.                     # all keypoints are displayed as crosses
    cross_thickness_img_size_ratio = 250.               # all keypoints are displayed as crosses
    line_thickness_img_size_ratio = 300.                # correspondences are connected with this line
    perspective_map_thickness_img_size_ratio = 200.     # homography is displayed as a polygon
    perspective_line_thickness_img_size_ratio = 150     # homography is displayed as a polygon
    inlier_color = (102., 255., 51.)                    # green - color of crosses and lines
    outlier_color = (255., 26., 26.)                    # red - color of crosses and lines
    kps_color = (179., 138., 77.)                       # dim orrange - color of keypoints
    perspective_map_color = (0., 0., 255.)              # blue - color of the homography polygon
    best_perspective_map_color = (255., 255., 255.)     # white - color of the best homography so far
    position_based_color_left = True                    # if True left image keypoints are colored based on position
    draw_lafs = True                                    # if True, the features are drawn as ellipces.



    @staticmethod
    def ratio_to_size(img, ratio):
        """
        We want to keep sizes relative to the image size.
        This function defines how we obtain a value that we use in library visualization functions.
        """
        if len(img.shape) == 3:
            return int(min(img.shape[:2]) / ratio)
        return int(min(img.shape) / ratio)


def draw_cross(img, center, color, thickness, d):
    cv2.line(img, (center[0] - d, center[1] - d), (center[0] + d, center[1] + d), color, thickness, cv2.LINE_AA, 0)
    cv2.line(img, (center[0] + d, center[1] - d), (center[0] - d, center[1] + d), color, thickness, cv2.LINE_AA, 0)
    return img


def add_offset(img_left, pts):
    """
    pts.shape = N, 4
    img_left is the left image -> to get the offset size
    """
    if not isinstance(pts, torch.Tensor):
        pts = torch.from_numpy(pts)
    if pts is not None and len(pts) > 0:
        return pts + torch.tensor([0, 0, img_left.shape[1], 0])
    return pts


def draw_crosses(img, pts: torch.Tensor, color, plot_params, colors = None):
    """ pts.shape = N, 2 """
    min_side = min(img.shape)
    cross_size = PlotParams.ratio_to_size(img, plot_params.cross_size_img_size_ratio)
    thickness = PlotParams.ratio_to_size(img, plot_params.cross_thickness_img_size_ratio)
    if not isinstance(pts, torch.Tensor):
        pts = torch.from_numpy(pts)
    for i, pt in enumerate(pts.round().to(dtype=torch.int32)):
        if colors is None:
            current_color  = color
        else:
            current_color = colors[i]
        img = draw_cross(img, pt.detach().cpu().numpy(), current_color, thickness, cross_size)
    return img


def draw_ellipse(img, contour, color, thickness):
    img = cv2.polylines(img, contour, True, color, thickness)
    return img


def draw_ellipses(img, pts: torch.Tensor, color, plot_params, colors = None):
    """ pts.shape = N, 2 """
    min_side = min(img.shape)
    cross_size = PlotParams.ratio_to_size(img, plot_params.cross_size_img_size_ratio)
    thickness = PlotParams.ratio_to_size(img, plot_params.cross_thickness_img_size_ratio)
    if not isinstance(pts, torch.Tensor):
        pts = torch.from_numpy(pts)
    boundary_pts = KF.laf_to_boundary_points(pts, 30).reshape(-1, 30, 2)

    for i, contour in enumerate(boundary_pts):
        if colors is None:
            current_color  = color
        else:
            current_color = colors[i]
        img = draw_ellipse(img, contour.unsqueeze(0).detach().cpu().numpy().astype(np.int32), current_color, thickness)
    return img


def draw_lines(img, pts: torch.Tensor, color, plot_params):
    """ pts.shape = N, 4 """
    thickness = PlotParams.ratio_to_size(img, plot_params.line_thickness_img_size_ratio)
    for pt in pts.round().to(dtype=torch.int32):
        pt = pt.detach().cpu().numpy()
        img = cv2.line(img, tuple(pt[:2]), tuple(pt[2:]), color, thickness)
    return img


def join_images(img1, img2):
    draw_params = dict(matchColor=(255, 255, 0),
                       singlePointColor=None,
                       matchesMask=[],
                       flags=20)
    return cv2.drawMatches(img1, [], img2, [], [], None, **draw_params)


def draw_perspective_map(img1, img, H, color, plot_params: PlotParams):
    if H is not None:
        h, w, ch = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        thickness = PlotParams.ratio_to_size(img, plot_params.perspective_line_thickness_img_size_ratio)
        if isinstance(H, torch.Tensor):
            H = H.detach().cpu().numpy()
        dst = cv2.perspectiveTransform(pts, H)
        res = cv2.polylines(img, [np.int32(dst)], True, color, thickness, cv2.LINE_AA)
        return res
    return img


def decolorize(img):
    """ de-colorizes the image and KEEPS the dimensions! """
    assert len(img.shape) in {2, 3}, f"Image is not in correct shape (H, W, Optional[C]), shape: {img.shape}"
    res = img.copy()
    if len(img.shape) == 3:
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)


def draw_keypoints(img1, img2, kps1, kps2, plt_params: PlotParams):
    imga, imgb = decolorize(img1), decolorize(img2)  # RGB -> GRAY -> GRAY RGB
    colors = None
    if not plt_params.draw_lafs:
        imga = draw_crosses(imga, KF.get_laf_center(kps1).view(-1, 2), plt_params.kps_color, plt_params, colors=colors)  # keypoints
        imgb = draw_crosses(imgb, KF.get_laf_center(kps2).view(-1, 2), plt_params.kps_color, plt_params, colors=colors)  # keypoints
    else:
        imga = draw_ellipses(imga, kps1, plt_params.kps_color, plt_params, colors=colors)  # keypoints
        imgb = draw_ellipses(imgb, kps2, plt_params.kps_color, plt_params, colors=colors)  # keypoints
    imgab = join_images(imga, imgb)
    return imgab


def draw_matches(img1, img2, detections, plt_params: PlotParams,
                 visualization_options: dict):
    """
    @param visualization_options: a dictionary that defines how to display tentative correspondences.
                                  Modes works as follows: there are 3 modes 0, 1, 2, 3
        #                         Mode 0: do not plot anything
        #                         Mode 1: show only keypoints (in both images)
        #                         Mode 2: show regions (in both images)
        #                         Mode 3: show keypoints and connect them with lines
        #                         Mode 4: show regions and connect them with lines
    """
    # todo: pts_matches none, mask None etc etc
    imga, imgb = decolorize(img1), decolorize(img2)  # RGB -> GRAY -> GRAY RGB
    #
    #Detections = namedtuple('Detections', ['kps1',
#                                       'kps2',
    #                                   'tentative_matches',
    #                                   'pts_matches',
    #                                   'H',
    #                                   'inlier_mask',
    #                                   'lafs1',
    #                                   'lafs2'])

    idxs = detections.tentative_matches
    tents_1 =  detections.lafs1[0:1,idxs[:, 0]]
    tents_2 =  detections.lafs2[0:1,idxs[:, 1]]
    inlier_mask = detections.inlier_mask.reshape(-1).bool()
    assert len(inlier_mask) == tents_1.shape[1], f"pts_matches ({len(tents_1)}) and mask ({len(inlier_mask)}) lengths differ"
    if len(inlier_mask) == 0:
        return
    if plt_params.position_based_color_left:
        colors_out = []
        for i, pt in enumerate(KF.get_laf_center(tents_1).view(-1, 2)):
            current_color = (255 * (pt[1].item() /imga.shape[0]),
                             0,
                             255 * (pt[0].item() /imga.shape[1]))
            colors_out.append(current_color)
    else:
        colors_out = None

    # drawing outliers first

    if visualization_options['outliers_display_mode'] in [1, 3]:
        imga = draw_crosses(imga, KF.get_laf_center(tents_1).view(-1, 2)[~inlier_mask], plt_params.kps_color, plt_params, colors=colors_out)  # keypoints
        imgb = draw_crosses(imgb, KF.get_laf_center(tents_2).view(-1, 2)[~inlier_mask], plt_params.kps_color, plt_params, colors=colors_out)  # keypoints
    if visualization_options['outliers_display_mode'] in [2, 4]:
        imga = draw_ellipses(imga, tents_1[:,~inlier_mask], plt_params.kps_color, plt_params, colors=colors_out)  # keypoints
        imgb = draw_ellipses(imgb, tents_2[:,~inlier_mask], plt_params.kps_color, plt_params, colors=colors_out)  # keypoints
    if visualization_options['inliers_display_mode'] in [1, 3]:
        imga = draw_crosses(imga, KF.get_laf_center(tents_1).view(-1, 2)[inlier_mask], plt_params.kps_color, plt_params, colors=colors_out)  # keypoints
        imgb = draw_crosses(imgb, KF.get_laf_center(tents_2).view(-1, 2)[inlier_mask], plt_params.kps_color, plt_params, colors=colors_out)  # keypoints
    if visualization_options['inliers_display_mode'] in [2, 4]:
        imga = draw_ellipses(imga, tents_1[:,inlier_mask], plt_params.kps_color, plt_params, colors=colors_out)  # keypoints
        imgb = draw_ellipses(imgb, tents_2[:,inlier_mask], plt_params.kps_color, plt_params, colors=colors_out)  # keypoints



    if detections.H is not None:    # best so far perspective map
        imgb = draw_perspective_map(imga, imgb, detections.H, plt_params.perspective_map_color, plt_params)  # perspective map
    # ----------- imgab
    pts_matches = torch.cat([KF.get_laf_center(tents_1).view(-1, 2),
                             KF.get_laf_center(tents_2).view(-1, 2)], dim=1)
    pts_offset = add_offset(img1, pts_matches)
    imgab = join_images(imga, imgb)
    if visualization_options['outliers_display_mode'] in [3, 4]:
        imgab = draw_lines(imgab, pts_offset[~inlier_mask], plt_params.outlier_color, plt_params)  # outliers connections
    if visualization_options['inliers_display_mode']  in [3, 4]:
        imgab = draw_lines(imgab, pts_offset[inlier_mask], plt_params.inlier_color, plt_params)  # inliers connections
    return imgab


def plot_legend_init(kps1, kps2, plt_params, fontsize=13):
    plt.rc('legend', fontsize=fontsize)
    fig = plt.figure(figsize=(.01, .01))
    ax = fig.add_subplot(111)
    # legend_items = [Tuple[handle, label]] -> I do this to ensure the correct order
    legend_items = [
        (ax.scatter([], [], color=np.array(plt_params.kps_color)/255., marker='x'), f'keypoints1: {len(kps1)}'),
        (ax.scatter([], [], color=np.array(plt_params.kps_color)/255., marker='x'), f'keypoints2: {len(kps2)}'),
    ]
    handles, labels = list(zip(*legend_items))
    plt.axis('off')
    ax.legend(handles, labels, loc='center')
    plt.show()
    return handles, labels


def plot_best_legend(log, detections, plt_params, fontsize=13):
    plt.rc('legend', fontsize=fontsize)
    fig = plt.figure(figsize=(.01, .01))
    ax = fig.add_subplot(111)
    legend_items = [
        (ax.plot([], [], ' ')[0], "$\\bf{best\ so\ far}$" + f" - i: {log.i + 1}"),
        (ax.plot([], [], ' ')[0], f'inliers: {detections.inlier_mask.long().sum().item()}'),
        (ax.plot([], [], ' ')[0], f'outliers: {len(detections.inlier_mask.reshape(-1)) - detections.inlier_mask.long().sum().item()}'),
        (ax.plot([], [], ' ')[0], f'loss: {log.loss:.2f}'),
        (Patch(facecolor=np.array(plt_params.best_perspective_map_color)/255., edgecolor='black'), 'perspective map'),
    ]
    handles, labels = list(zip(*legend_items))
    plt.axis('off')
    ax.legend(handles, labels, loc='center')
    plt.show()
    return handles, labels


def plot_current_legend(log, detections, plt_params, fontsize=13):
    plt.rc('legend', fontsize=fontsize)
    fig = plt.figure(figsize=(.01, .01))
    ax = fig.add_subplot(111)
    # legend_items = [Tuple[handle, label]] -> I do this to ensure the correct order
    legend_items = [
        (ax.plot([], [], ' ')[0], "$\\bf{current}$" + f" - i: {log.i + 1}"),
        (ax.scatter([], [], color=np.array(plt_params.inlier_color)/255., marker='x'),
            f'inliers: {detections.inlier_mask.long().sum().item()}'),
        (ax.scatter([], [], color=np.array(plt_params.outlier_color)/255., marker='x'),
            f'outliers: {len(detections.inlier_mask.reshape(-1)) - detections.inlier_mask.long().sum().item()}'),
        (ax.plot([], color=np.array(plt_params.inlier_color)/255.)[0], f'loss: {log.loss:.2f}'),
        (Patch(facecolor=np.array(plt_params.perspective_map_color)/255., edgecolor='black'), 'perspective map'),
    ]
    handles, labels = list(zip(*legend_items))
    plt.axis('off')
    ax.legend(handles, labels, loc='center')
    plt.show()
    return handles, labels


def cv2_draw_matches(kps1, kps2, tentative_matches, H, inlier_mask, img1, img2, H_gt = None):
    """
    detections = Detections(*detect_describe_match(timg1, timg2, params, functions))
    log = ransac_planar(detections.pts_matches, params, functions)
    imgout = cv2_draw_matches(detections.kps1, detections.kps2, detections.tentative_matches,
             log.H, log.mask, decolorize(img1), decolorize(img2))
    """
    matches_mask = inlier_mask.ravel().tolist()
    h, w, ch = img1.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)
    img2_tr = cv2.polylines(decolorize(img2), [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
    if H_gt is not None:        # Ground truth transformation
        dst_gt = cv2.perspectiveTransform(pts, H_gt)
        img2_tr = cv2.polylines(deepcopy(img2_tr), [np.int32(dst_gt)], True, (0, 255, 0), 3, cv2.LINE_AA)
    # Blue is estimated, green is ground truth homography
    draw_params = dict(matchColor=(255, 255, 0), singlePointColor=None,
                       matchesMask=matches_mask, flags=20)
    img_out = cv2.drawMatches(decolorize(img1), kps1, img2_tr, kps2, tentative_matches, None, **draw_params)
    plt.figure(figsize=(20, 10))
    plt.imshow(img_out)
    plt.show()
    return img_out
