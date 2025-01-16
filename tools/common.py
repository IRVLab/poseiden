import cv2
import numpy as np
import torch


def fix_aspect_ratio(bbox_scale, aspect_ratio):
    """Reshape the bbox to a fixed aspect ratio.

    Args:
        bbox_scale (np.ndarray): The bbox scales (w, h) in shape (n, 2)
        aspect_ratio (float): The ratio of ``w/h``

    Returns:
        np.darray: The reshaped bbox scales in (n, 2)
    """

    w, h = np.hsplit(bbox_scale, [1])
    bbox_scale = np.where(w > h * aspect_ratio,
                          np.hstack([w, w / aspect_ratio]),
                          np.hstack([h * aspect_ratio, h]))
    return bbox_scale


def rotate_point(pt, angle_rad):
    """Rotate a point by an angle.

    Args:
        pt (np.ndarray): 2D point coordinates (x, y) in shape (2, )
        angle_rad (float): rotation angle in radian

    Returns:
        np.ndarray: Rotated point in shape (2, )
    """

    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt


def get_3rd_point(a, b):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): The 1st point (x,y) in shape (2, )
        b (np.ndarray): The 2nd point (x,y) in shape (2, )

    Returns:
        np.ndarray: The 3rd point.
    """
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c


def get_warp_matrix(center, scale, rot, output_size,
                    shift=(0., 0.), inv=False, fix_aspect_ratio=True):
    """Calculate the affine transformation matrix that can warp the bbox area
    in the input image to the output size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)
        fix_aspect_ratio (bool): Whether to fix aspect ratio during transform.
            Defaults to True.

    Returns:
        np.ndarray: A 2x3 transformation matrix
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    shift = np.array(shift)
    src_w, src_h = scale[:2]
    dst_w, dst_h = output_size[:2]

    rot_rad = np.deg2rad(rot)
    src_dir = rotate_point(np.array([src_w * -0.5, 0.]), rot_rad)
    dst_dir = np.array([dst_w * -0.5, 0.])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    if fix_aspect_ratio:
        src[2, :] = get_3rd_point(src[0, :], src[1, :])
        dst[2, :] = get_3rd_point(dst[0, :], dst[1, :])
    else:
        src_dir_2 = rotate_point(np.array([0., src_h * -0.5]), rot_rad)
        dst_dir_2 = np.array([0., dst_h * -0.5])
        src[2, :] = center + src_dir_2 + scale * shift
        dst[2, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir_2

    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return warp_mat


def world_to_camera(points, R, T):
    # Construct the transformation matrix
    Rt = np.concatenate((R, T), axis=1)
    Rt = np.concatenate((Rt, np.array([[0, 0, 0, 1]])), axis=0)

    # Transform the 3D points into the camera coordinate system
    points_hom = np.vstack((points.T,
                            np.ones((1, points.shape[0]))))

    points_hom = Rt @ points_hom

    return points_hom[:3].T


def camera_to_image(points, K):
    points_2d = K @ points.T

    points_2d = points_2d.T
    points_2d[:, :2] /= points_2d[:, 2:]

    return points_2d


def get_projection_matrix(K, R, T):
    P = K @ np.hstack((R, T))
    P = np.vstack((P, np.array([0, 0, 0, 1])))

    return P


def project_3d_to_2d(pose_3d, K, R, T):
    # Transform the 3D points into the camera coordinate system
    pose_2d = world_to_camera(pose_3d, R, T)
    pose_2d = camera_to_image(pose_2d, K)

    return pose_2d


def undistort_image(image, K, dist_coeffs, new_K=np.array([])):
    if new_K.size == 0:
        new_K = K.copy()
    undistorted_image = cv2.undistort(image, K, dist_coeffs, None, new_K)

    return undistorted_image


def triangulation(P1, P2, pts1, pts2):
    pts3D = []

    for pt1, pt2 in zip(pts1, pts2):
        pt1_ = np.array([pt1[0], pt1[1], 1])
        pt2_ = np.array([pt2[0], pt2[1], 1])

        pt1_ = np.cross(pt1_, np.identity(pt1_.shape[0]) * -1)
        pt2_ = np.cross(pt2_, np.identity(pt2_.shape[0]) * -1)

        M1 = np.array([pt1[1] * P1[2] - P1[1], P1[0] - pt1[0] * P1[2]])
        M2 = np.array([pt2[1] * P2[2] - P2[1], P2[0] - pt2[0] * P2[2]])
        M = np.vstack((M1, M2))

        e, v = np.linalg.eig(M.T @ M)
        idx = np.argmin(e)
        pt3 = v[:, idx]
        pt3 = (pt3 / pt3[-1])[:3]
        pts3D.append(pt3)

    return np.array(pts3D)


def backproject(points, depths, proj_inv):
    ones = np.ones_like(points[..., :1])

    points = np.concatenate([points, ones], axis=-1)
    points = points * depths

    points = np.concatenate([points, ones], axis=-1)

    # project from image plane to world coordinate system
    points = np.matmul(proj_inv, points.T).T

    points = points[..., :3] / points[..., 3:]

    return points


def project(points, proj):
    ones = np.ones_like(points[..., :1])

    points = np.concatenate([points, ones], axis=-1)

    # project from world coordinate system to image plane
    points = np.matmul(proj, points.T).T

    points = points[..., :2] / points[..., 2:3]

    return points


def calc_3d_joint(kps_2d, disp, baseline, proj_list):
    proj_inv_list = [np.linalg.inv(proj) for proj in proj_list]

    # The [0, 0] location of P matrix is equal to the focal length of the
    # camera after rectification
    focal_l = proj_list[0][0:1, 0:1]
    focal_r = proj_list[1][0:1, 0:1]

    # compute depth from disparity
    depth_l = focal_l * baseline / (disp + 1e-15)
    depth_r = focal_r * baseline / (disp + 1e-15)

    # project 2D keypoints to left camera space using inverse
    # projection matrix
    kp_3d_left = backproject(kps_2d[0], depth_l, proj_inv_list[0])
    kp_3d_right = backproject(kps_2d[1], depth_r, proj_inv_list[1])

    kp_3d = (kp_3d_left + kp_3d_right) / 2

    return kp_3d


def reproject(kpt_2d, disp, Q):
    """
    Reproject 2D keypoints to 3D using disparity

    Args:
        kpt_2d: 2D keypoints [batch_size, N, 2]
        disp: disparity [batch_size, N, 1]
        Q: disparity-to-depth mapping matrix [batch_size, 4, 4]
    """
    kpt_2d_ = kpt_2d.transpose(0, 2, 1)
    disp_ = disp.transpose(0, 2, 1)

    kpt_3d = np.concatenate((kpt_2d_, disp_, np.ones_like(disp_)), axis=1)
    kpt_3d = np.matmul(Q, kpt_3d).transpose(0, 2, 1)

    return kpt_3d[..., :3] / kpt_3d[..., 3:]


def project_tensor(points, proj):
    ones = torch.ones_like(points[..., :1])

    points = torch.cat([points, ones], dim=-1)

    # project from world coordinate system to image plane
    points = torch.matmul(proj, points.transpose(1, 2)).transpose(1, 2)

    points = points[..., :2] / points[..., 2:3]

    return points


def reproject_tensor(kpt_2d, disp, Q):
    """
    Reproject 2D keypoints to 3D using disparity

    Args:
        kpt_2d: 2D keypoints [batch_size, N, 2]
        disp: disparity [batch_size, N, 1]
        Q: disparity-to-depth mapping matrix [batch_size, 4, 4]
    """
    kpt_2d_ = kpt_2d.transpose(1, 2)
    disp_ = disp.transpose(1, 2)

    kpt_3d = torch.cat((kpt_2d_, disp_, torch.ones_like(disp_)), dim=1)
    kpt_3d = torch.matmul(Q, kpt_3d).transpose(1, 2)

    return kpt_3d[..., :3] / kpt_3d[..., 3:]


def gen_reproj_matrix(P_left, P_right, baseline):
    """
    Create a disparity-to-depth mapping matrix.
    Identical to the Q matrix from cv2.stereoRectify()

    Args:
        P_left: camera projection matrix for left image
        P_right: camera projection matrix for right image
        baseline: distance between the two cameras
    Returns:
        Q: disparity-to-depth mapping matrix
    """

    Q = np.eye(4)
    Q[0, 3] = -P_left[0, 2]
    Q[1, 3] = -P_left[1, 2]
    Q[2, 3] = P_left[0, 0]
    Q[2, 2] = 0
    Q[3, 2] = 1 / baseline
    Q[3, 3] = -(P_left[0, 2] - P_right[0, 2]) / baseline

    return Q
