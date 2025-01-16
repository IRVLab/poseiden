import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial.transform import Rotation
import torch
import cv2
import os
import math
import onnxruntime as ort
from sklearn.metrics import (ConfusionMatrixDisplay, precision_score,
                             recall_score)

matplotlib.use("Agg")


def to_torch(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        x = x.type(torch.float32)

    return x


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    return x


def fig2array(fig):
    canvas = fig.canvas
    canvas.draw()
    width, height = canvas.get_width_height()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_array = image_array.reshape(height, width, 3)
    return image_array


def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def plot_body(ax, points, color, label, visibility=None, diver=False):

    # Define the connections between the joints
    if diver:  # COCO format
        connections = [
            (0, 2), (2, 4),  # left arm
            (1, 3), (3, 5),  # right arm
            (6, 8), (8, 10),  # left leg
            (7, 9), (9, 11),  # right leg
            (0, 6), (1, 7), (0, 1), (6, 7),  # body
        ]
    else:
        connections = [
            (0, 1),   # body
            (0, 18),  # head
            (1, 6), (6, 7), (7, 8), (8, 9),  # left leg
            (0, 14), (14, 15), (15, 16), (16, 17),  # left arm
            (1, 2), (2, 3), (3, 4), (4, 5),  # right leg
            (0, 10), (10, 11), (11, 12), (12, 13),  # right arm
        ]

    # Plot the skeleton joints
    if visibility is not None:
        ax.scatter(points[visibility, 0],
                   points[visibility, 1],
                   points[visibility, 2],
                   c=color, marker='o', s=2)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=color, marker='o', s=2)

    for connection in connections:
        joint1 = points[connection[0]]
        joint2 = points[connection[1]]
        if visibility is not None:
            if visibility[connection[0]] and visibility[connection[1]]:
                ax.plot([joint1[0], joint2[0]],
                        [joint1[1], joint2[1]],
                        [joint1[2], joint2[2]], c=color)
            else:
                ax.plot([joint1[0], joint2[0]],
                        [joint1[1], joint2[1]],
                        [joint1[2], joint2[2]], c='gray')
        else:
            ax.plot([joint1[0], joint2[0]],
                    [joint1[1], joint2[1]],
                    [joint1[2], joint2[2]], c=color)

    ax.plot([], [], c=color, label=label)


def plot_axis(ax, c, x, y, z):
    ax.quiver(*c, *x, color='r', length=1000,
              arrow_length_ratio=0.1, label='X')
    ax.quiver(*c, *y, color='g', length=1000,
              arrow_length_ratio=0.1, label='Y')
    ax.quiver(*c, *z, color='b', length=1000,
              arrow_length_ratio=0.1, label='Z')


def plot_pose_3d(pose_3d, pred_3d, axis=None, visibility=None,
                 xlim=[-500, 1500], ylim=[0, 3000], zlim=[0, 1500],
                 diver=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(xlim)
    ax.set_ylim3d(ylim)
    ax.set_zlim3d(zlim)

    # Set the view to bird-eye view
    # ax.view_init(elev=90, azim=-90)

    rot = Rotation.from_euler('zyx', np.array([0, 0, -90]),
                              degrees=True).as_matrix()

    if pose_3d is not None:
        pose_3d = (rot @ pose_3d.T).T
        plot_body(ax, pose_3d, 'r', "ground truth", visibility, diver)

    if pred_3d is not None:
        pred_3d = (rot @ pred_3d.T).T
        plot_body(ax, pred_3d, 'g', "estimation", visibility, diver)

    if axis:
        c, x, y, z = axis
        c = (rot @ c.T).T
        x = (rot @ x.T).T
        y = (rot @ y.T).T
        z = (rot @ z.T).T

        # Plot the axes
        plot_axis(ax, c, x, y, z)

    # Set labels and title
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('3D Human Skeleton')
    ax.legend()

    # ax.zaxis.set_ticks([])

    fig.tight_layout()

    # Convert the plot to numpy array
    image_array = fig2array(fig)
    plt.close()

    return image_array


def plot_joints(img, joints, visibility=None, c=(0, 255, 0)):
    for k in range(joints.shape[0]):
        color = c

        if visibility is not None:
            if not visibility[k]:
                color = (0, 0, 255)

        joint = joints[k]
        cv2.circle(img, (int(joint[0]), int(joint[1])),
                   2, color, -1)

    return img


def plot_pose_2d(gt_joints, pred_joints, imgs, visibility):
    for gt, pred, img in zip(gt_joints, pred_joints, imgs):
        img = plot_joints(img, gt, visibility, c=(0, 0, 255))
        img = plot_joints(img, pred, c=(0, 255, 0))

    img = np.concatenate(imgs, axis=1)

    return img


def plot_error(error, frame_num):
    fig = plt.figure(figsize=(5.12, 2.24))
    ax = fig.add_subplot(111)
    ax.set_xticks(list(range(0, frame_num+1, 1000)))
    ax.set_yticks(list(range(0, 301, 100)))
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xlim(0, frame_num)
    ax.set_ylim(0, 300)
    ax.plot(error)

    ax.set_xlabel('Frame', fontsize=8)
    ax.set_ylabel('Error (mm)', fontsize=8)
    ax.set_title('MPJPE', size=10)

    fig.tight_layout()

    image_array = fig2array(fig)
    plt.close()

    return image_array


def analyze_accuracy(targ_list, pred_list, labels=None):
    precision = precision_score(targ_list, pred_list,
                                average='macro', zero_division=0)
    recall = recall_score(targ_list, pred_list,
                          average='macro', zero_division=0)

    cm_display = ConfusionMatrixDisplay.from_predictions(
        targ_list, pred_list, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 8))
    cm_display.plot(include_values=True, cmap='Blues', ax=ax)
    # Remove extra white space around the plot
    plt.tight_layout()
    plt.xticks(rotation=90)

    image_array = fig2array(fig)
    plt.close()

    return precision, recall, image_array


def compute_diver_body_frame(pose_3d):
    """
    Compute the diver body frame given the 3D pose.

    Args:
        pose_3d : numpy.ndarray (N, 3)
            The 3D pose of the diver.

    Returns:
        center_mass : numpy.ndarray (3,)
            The center of mass of the diver's body frame.
        x_hat : numpy.ndarray (3,)
            The x-axis unit vector of the diver's body frame.
        y_hat : numpy.ndarray (3,)
            The y-axis unit vector of the diver's body frame.
        z_hat : numpy.ndarray (3,)
            The z-axis unit vector of the diver's body frame.
    """
    r_shoulder, l_shoulder = pose_3d[1], pose_3d[0]
    r_hip, l_hip = pose_3d[7], pose_3d[6]

    center_mass = np.mean(
        [r_shoulder, l_shoulder, r_hip, l_hip], axis=0)

    rhs = r_shoulder - r_hip
    rhls = l_shoulder - r_hip
    lhs = l_shoulder - l_hip
    lhrs = r_shoulder - l_hip

    r_cross = np.cross(rhls, rhs)
    l_cross = np.cross(lhs, lhrs)

    z = np.vstack((r_cross, l_cross)).mean(axis=0)
    z_hat = z / (np.linalg.norm(z) + 1e-5)

    hip_midpt = (r_hip + l_hip) / 2
    y = hip_midpt - center_mass
    y_hat = y / (np.linalg.norm(y) + 1e-5)

    # x is the cross product of y_hat and z_hat
    x_hat = np.cross(y_hat, z_hat)

    return center_mass, x_hat, y_hat, z_hat


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True,
              scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    # wh padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        # width, height ratios
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def check_img_size(img_size, s=32):
    def make_divisible(x, divisor):
        # Returns x evenly divisible by divisor
        return math.ceil(x / divisor) * divisor
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of " \
              "max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size


def scale_coords(img1_shape, coords, img0_shape,
                 ratio_pad=None, kpt_label=False, step=2):
    def clip_coords(boxes, img_shape, step=2):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0::step].clip(0, img_shape[1])  # x1
        boxes[:, 1::step].clip(0, img_shape[0])  # y1
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2,
               (img1_shape[0] - img0_shape[0] * gain) / 2)  # wh padding
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]
    if isinstance(gain, (list, tuple)):
        gain = gain[0]

    if not kpt_label:
        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, [0, 2]] /= gain
        coords[:, [1, 3]] /= gain
        clip_coords(coords[0:4], img0_shape)
    else:
        coords[:, 0::step] -= pad[0]  # x padding
        coords[:, 1::step] -= pad[1]  # y padding
        coords[:, 0::step] /= gain
        coords[:, 1::step] /= gain
        clip_coords(coords, img0_shape, step=step)
    return coords


def load_onnx_model(weight_path):
    if not os.path.exists(weight_path):
        assert False, "Model is not exist in {}".format(weight_path)

    session = ort.InferenceSession(
        weight_path,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    print("ONNX is using {}".format(ort.get_device()))

    return session
