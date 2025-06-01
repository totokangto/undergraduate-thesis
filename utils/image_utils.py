#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

# 0 제외
def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def extract_valid_motion_pixels(
    image, gt_image, gradient_threshold=5, magnitude_threshold=1.0):

    # 텐서를 NumPy로 변환 및 HWC 변환
    image = image.permute(1, 2, 0).cpu().numpy()
    gt_image = gt_image.permute(1, 2, 0).cpu().numpy()

    # 0~1 범위일 경우 0~255로 변환
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
        gt_image = (gt_image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
        gt_image = gt_image.astype(np.uint8)


    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_gt = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    flow = dis.calc(gray_img, gray_gt, None)

    grad_x = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    texture_mask = grad_mag > gradient_threshold

    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    motion_mask = mag > magnitude_threshold

    combined_mask = texture_mask & motion_mask
    np_image = np.zeros_like(image)
    np_gt = np.zeros_like(gt_image)
    np_image[combined_mask] = image[combined_mask]
    np_gt[combined_mask] = gt_image[combined_mask]

    pixel_image = torch.from_numpy(np_image.copy().astype(np.float32) / 255.0).permute(2, 0, 1)
    pixel_gt = torch.from_numpy(np_gt.copy().astype(np.float32) / 255.0).permute(2, 0, 1)

    return pixel_image, pixel_gt, combined_mask


def visualize_optical_flow(
    image, gt_image, gradient_threshold=5, magnitude_threshold=1.0, flow_step=5, flow_scale=0.3
):
    # 텐서를 NumPy로 변환 및 HWC 변환
    image = image.permute(1, 2, 0).cpu().numpy()
    gt_image = gt_image.permute(1, 2, 0).cpu().numpy()

    # 0~1 범위일 경우 0~255로 변환
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
        gt_image = (gt_image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
        gt_image = gt_image.astype(np.uint8)

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_gt = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    flow = dis.calc(gray_img, gray_gt, None)

    grad_x = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    texture_mask = grad_mag > gradient_threshold

    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    motion_mask = mag > magnitude_threshold

    combined_mask = texture_mask & motion_mask

    h, w = gray_img.shape
    y_grid, x_grid = np.mgrid[flow_step//2:h:flow_step, flow_step//2:w:flow_step]
    valid_points = combined_mask[y_grid, x_grid]
    x_valid = x_grid[valid_points]
    y_valid = y_grid[valid_points]
    fx = flow[..., 0][y_grid, x_grid][valid_points]
    fy = flow[..., 1][y_grid, x_grid][valid_points]

    background = image.copy()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(background)
    ax.quiver(x_valid, y_valid, fx, fy, color='cyan', angles='xy', scale_units='xy', scale=flow_scale)
    ax.set_title(f'Optical Flow Visualization (Tensor Output, step={flow_step})')
    ax.axis('off')
    plt.tight_layout()

    fig.canvas.draw()
    vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    vis_tensor = torch.from_numpy(vis_image).permute(2, 0, 1).float() / 255.0
    return vis_tensor
