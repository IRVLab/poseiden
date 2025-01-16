import cv2
import numpy as np


def fliplr(image, joints, joints_vis, width, matched_parts):
    """Flip the image and joints horizontally.
    Args:
        image: input image
        joints: [num_joints, 2 or 3]
        joints_vis: [num_joints, 1]
        width: image width
        matched_parts: pairs of joints to flip
    """
    assert joints.shape[0] == joints_vis.shape[0], \
        'joints and joints_vis should have the same number of joints, ' \
        'current shape is joints={}, joints_vis={}'.format(
            joints.shape, joints_vis.shape)

    # Flip horizontal
    image = image[:, ::-1, :]
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return image, joints, joints_vis


def flipud(image, joints, joints_vis, height, matched_parts):
    """Flip the image and joints horizontally.
    Args:
        image: input image
        joints: [num_joints, 2 or 3]
        joints_vis: [num_joints, 1]
        width: image width
        matched_parts: pairs of joints to flip
    """
    assert joints.shape[0] == joints_vis.shape[0], \
        'joints and joints_vis should have the same number of joints, ' \
        'current shape is joints={}, joints_vis={}'.format(
            joints.shape, joints_vis.shape)

    # Flip horizontal
    image = image[::-1, :, :]
    joints[:, 1] = height - joints[:, 1] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return image, joints, joints_vis


def hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    """HSV color-space augmentation.
    Args:
        img: input image
        hgain: random gains for hue
        sgain: random gains for saturation
        vgain: random gains for value
    """
    if hgain or sgain or vgain:
        # random gains
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue),
                            cv2.LUT(sat, lut_sat),
                            cv2.LUT(val, lut_val)))
        img = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)

    return img


def hidenseek(image, joints, joints_vis, max_grid=20, grid_size=15):
    """Hide and Seek augmentation strategy.
    Args:
        image: input image
        joints: [num_joints, 2]
        joints_vis: [num_joints, 2]
        max_grid: maximum number of grid points to hide
        grid_size: size of the grid
    """
    # create a bounding box from the max and min values in joints
    x_min = np.min(joints[joints_vis[:, 0]][:, 0]).astype(int)
    y_min = np.min(joints[joints_vis[:, 1]][:, 1]).astype(int)
    x_max = np.max(joints[joints_vis[:, 0]][:, 0]).astype(int)
    y_max = np.max(joints[joints_vis[:, 1]][:, 1]).astype(int)

    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max, grid_size),
        np.arange(y_min, y_max, grid_size)
    )

    x_f = np.ravel(grid_x)
    y_f = np.ravel(grid_y)
    grid_pts = np.array(list(zip(x_f, y_f)))

    n_choices = np.random.randint(1, max_grid + 1)

    if n_choices < len(grid_pts) * 0.5:
        random_ind = np.random.choice(len(grid_pts), n_choices, replace=False)
        random_pts = grid_pts[random_ind]

        # according to the paper, this ensure the activations to have a
        # distribution that is same to those seen during testing
        mean_pixel_val = np.mean(image, axis=(0, 1))

        for pts in random_pts:
            x, y = pts
            image[y:y + grid_size, x:x + grid_size, :] = mean_pixel_val

    return image
