import numpy as np
import torch.nn.functional as F
import torchvision
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from einops import rearrange

from .utils import get_max_preds, to_numpy


def norm_image(image):
    min = float(image.min())
    max = float(image.max())
    image.add_(-min).div_(max - min + 1e-5)
    return image


def norm_atten_map(atten_map):
    atten_map = (
        (atten_map - np.min(atten_map))
        / (np.max(atten_map) - np.min(atten_map))
    )
    return atten_map


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    file_name: saved file name
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])),
                               2, [255, 0, 0], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = norm_image(batch_image)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images(input, meta, target, joints_pred, output, prefix):
    save_batch_image_with_joints(
        input, meta['joints'], meta['joints_vis'],
        '{}_gt.jpg'.format(prefix)
    )
    save_batch_image_with_joints(
        input, joints_pred, meta['joints_vis'],
        '{}_pred.jpg'.format(prefix)
    )
    save_batch_heatmaps(
        input, target, '{}_hm_gt.jpg'.format(prefix)
    )
    save_batch_heatmaps(
        input, output, '{}_hm_pred.jpg'.format(prefix)
    )


def torch_img_to_bgr_img(image):
    # convert tensor to numpy
    image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0)
    image = to_numpy(image)

    # resize image and convert RGB to BGR
    resized_image = cv2.resize(
        image.copy(), (image.shape[1]//4, image.shape[0]//4))
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)

    return resized_image


def save_mono_attention_map(
        batch_image, batch_joints, batch_attn, file_name,
        normalize_image=True, normalize_atten_map=True, num_display=16):

    batch_joints = to_numpy(batch_joints)

    # for now we assume the input image is square
    feat_size = int(math.sqrt(batch_attn.size(-1)))
    # average attention map over heads and reshape
    batch_attn = rearrange(
        batch_attn.mean(dim=1), 'b (h1 w1) (h2 w2) -> b h1 w1 h2 w2',
        h1=feat_size, w1=feat_size, h2=feat_size, w2=feat_size)

    if normalize_image:
        batch_image = norm_image(batch_image)

    num_joints = batch_joints.shape[1]

    fig = plt.figure(figsize=(30, 30))
    fig.subplots_adjust(
        bottom=0.02, right=0.97, top=0.98, left=0.03,
    )

    outer = gridspec.GridSpec(1, num_display, wspace=0.15)

    for b in range(num_display):
        image = torch_img_to_bgr_img(batch_image[b])
        attn_map = batch_attn[b]

        inner = gridspec.GridSpecFromSubplotSpec(
            num_joints + 1, 1,
            subplot_spec=outer[b],
            wspace=0.001, hspace=0.05)

        ax = plt.Subplot(fig, inner[0])

        ax.set_xlabel(f"sample_{b}", fontsize=20)
        ax.xaxis.set_label_position('top')
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)

        for j in range(num_joints):
            ax = plt.Subplot(fig, inner[j + 1])

            ax.imshow(image)

            pos = (batch_joints[b][j] / 16 + 0.5).astype(np.int32)
            pos = np.clip(pos, 0, feat_size - 1)

            attn_map_at_joint = F.interpolate(
                attn_map[None, None, pos[1], pos[0], :, :],
                scale_factor=4,
                mode="bilinear").squeeze()
            attn_map_at_joint = to_numpy(attn_map_at_joint)

            if normalize_atten_map:
                attn_map_at_joint = norm_atten_map(attn_map_at_joint)

            im = ax.imshow(
                attn_map_at_joint, cmap="nipy_spectral", alpha=0.7)

            pos = batch_joints[b][j] / 4
            ax.scatter(
                x=pos[0], y=pos[1], s=60, marker='*', c="white")
            ax.scatter(
                x=pos[0], y=pos[1], s=60, marker='*', c="white")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)

    cax = plt.axes([0.975, 0.025, 0.005, 0.95])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0.0, 0.5, 1])
    cb.ax.tick_params(labelsize=20)
    plt.savefig(file_name)
    plt.close()


def save_stereo_attention_map(
        batch_left_image, batch_right_image,
        batch_left_joints, batch_right_joints,
        batch_attn, file_name,
        normalize_image=True, normalize_atten_map=True,
        num_display=4):

    batch_left_joints = to_numpy(batch_left_joints)
    batch_right_joints = to_numpy(batch_right_joints)

    map_size = batch_attn.size(-1)
    # features in L that are similar to L
    L2L = batch_attn[..., :map_size//2, :map_size//2].mean(dim=1)
    # features in R that are similar to R
    R2R = batch_attn[..., map_size//2:, map_size//2:].mean(dim=1)
    # features in R that are similar to L
    L2R = batch_attn[..., :map_size//2, map_size//2:].mean(dim=1)
    # features in L that are similar to R
    R2L = batch_attn[..., map_size//2:, :map_size//2].mean(dim=1)

    batch_attn = {
        "self_L2L": L2L,
        "self_R2R": R2R,
        "self_L2R": L2R,
        "self_R2L": R2L,
    }

    # for now we assume the input image is square
    feat_size = int(math.sqrt(map_size//2))

    batch_attn = {
        name: rearrange(attn, 'b (h1 w1) (h2 w2) -> b h1 w1 h2 w2',
                        h1=feat_size, w1=feat_size, h2=feat_size, w2=feat_size)
        for name, attn in batch_attn.items()
    }

    if normalize_image:
        batch_left_image = norm_image(batch_left_image)
        batch_right_image = norm_image(batch_right_image)

    num_joints = batch_left_joints.shape[1]

    fig = plt.figure(figsize=(30, 30))
    fig.subplots_adjust(
        bottom=0.02, right=0.97, top=0.98, left=0.03,
    )

    outer = gridspec.GridSpec(1, num_display * 2, wspace=0.15)

    for b in range(num_display):
        left_image = torch_img_to_bgr_img(batch_left_image[b])
        right_image = torch_img_to_bgr_img(batch_right_image[b])

        image_list = [
            left_image, right_image, left_image, right_image]
        # for plotting joints on images
        joints_list_for_img = [
            batch_left_joints[b], batch_right_joints[b],
            batch_left_joints[b], batch_right_joints[b]]
        # for indexing attention map from joints
        joints_list_for_atten_map = [
            batch_left_joints[b], batch_right_joints[b],
            batch_right_joints[b], batch_left_joints[b]]

        for i, name in enumerate(batch_attn.keys()):
            attn_map = batch_attn[name][b]

            if i % 2 == 0:
                inner = gridspec.GridSpecFromSubplotSpec(
                    num_joints + 1, 2,
                    subplot_spec=outer[b * 2 + i // 2],
                    wspace=0.001, hspace=0.05)

            ax = plt.Subplot(fig, inner[i % 2])

            ax.set_xlabel("{}".format(name), fontsize=20)
            ax.xaxis.set_label_position('top')
            ax.imshow(image_list[i])
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)

            for j in range(num_joints):
                ax = plt.Subplot(fig, inner[(j + 1) * 2 + i % 2])

                ax.imshow(image_list[i])

                pos = (
                    joints_list_for_atten_map[i][j] / 16 + 0.5
                ).astype(np.int32)
                pos = np.clip(pos, 0, feat_size - 1)

                attn_map_at_joint = F.interpolate(
                    attn_map[None, None, pos[1], pos[0], :, :],
                    scale_factor=4,
                    mode="bilinear").squeeze()
                attn_map_at_joint = to_numpy(attn_map_at_joint)

                if normalize_atten_map:
                    attn_map_at_joint = norm_atten_map(attn_map_at_joint)

                im = ax.imshow(
                    attn_map_at_joint, cmap="nipy_spectral", alpha=0.7)

                pos = joints_list_for_img[i][j] / 4
                ax.scatter(
                    x=pos[0], y=pos[1], s=60, marker='*', c="white")
                ax.scatter(
                    x=pos[0], y=pos[1], s=60, marker='*', c="white")
                ax.set_xticks([])
                ax.set_yticks([])
                fig.add_subplot(ax)

    cax = plt.axes([0.975, 0.025, 0.005, 0.95])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([0.0, 0.5, 1])
    cb.ax.tick_params(labelsize=20)
    plt.savefig(file_name)
    plt.close()
