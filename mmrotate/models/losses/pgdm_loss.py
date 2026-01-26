# Copyright (c) OpenMMLab. All rights reserved.
"""
PGDM Loss (Pseudo Ground Truth Distribution Mask Loss)
Originally known as Voronoi Watershed Loss

This module provides mask generation and loss calculation for pseudo ground truth
using either SAM (Segment Anything Model) or Voronoi+Watershed approach.
"""
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
from mmrotate.registry import MODELS
from mmrotate.models.losses.vis import plot_gaussian_voronoi_watershed, visualize_loss_calculation, save_debug_visualization
from mmrotate.models.losses.utils import filter_masks
from mmdet.models.losses.utils import weighted_loss


@weighted_loss
def gwd_sigma_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, normalize=True):
    """Gaussian Wasserstein distance loss for sigma only.

    Args:
        pred (torch.Tensor): Predicted covariance matrices.
        target (torch.Tensor): Target covariance matrices.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.
        alpha (float): Defaults to 1.0.
        normalize (bool): Whether to normalize the distance. Defaults to True.

    Returns:
        loss (torch.Tensor)
    """
    from mmrotate.models.losses.gaussian_dist_loss import postprocess

    Sigma_p = pred
    Sigma_t = target

    whr_distance = Sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_distance = whr_distance + Sigma_t.diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    _t_tr = (Sigma_p.bmm(Sigma_t)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = (Sigma_p.det() * Sigma_t.det()).clamp(1e-7).sqrt()
    whr_distance = whr_distance + (-2) * (
        (_t_tr + 2 * _t_det_sqrt).clamp(1e-7).sqrt())

    distance = (alpha * alpha * whr_distance).clamp(1e-7).sqrt()

    if normalize:
        scale = 2 * (
            _t_det_sqrt.clamp(1e-7).sqrt().clamp(1e-7).sqrt()).clamp(1e-7)
        distance = distance / scale

    return postprocess(distance, fun=fun, tau=tau)


def gaussian_2d(xy, mu, sigma, normalize=False):
    """Calculate 2D Gaussian probability."""
    dxy = (xy - mu).unsqueeze(-1)
    t0 = torch.exp(-0.5 * dxy.permute(0, 2, 1).bmm(torch.linalg.solve(sigma, dxy)))
    if normalize:
        t0 = t0 / (2 * np.pi * sigma.det().clamp(1e-7).sqrt())
    return t0


def sigma_to_rbox_params(sigma: torch.Tensor):
    """Convert covariance matrix to rotation box parameters."""
    if not (sigma.shape == (2, 2)):
        raise ValueError("输入必须是一个 (2, 2) 的张量")
    L, V = torch.linalg.eigh(sigma)
    W_rotated = 2 * torch.sqrt(L[1])
    H_rotated = 2 * torch.sqrt(L[0])
    major_axis_vector = V[:, 1]
    angle_rad = torch.atan2(major_axis_vector[1], major_axis_vector[0])
    return W_rotated, H_rotated, angle_rad


def _get_box_prompt_from_gaussian(mu_j, sigma_j, sigma_scale=1, ellipse_scale_factor=1):
    """Get box prompt from Gaussian parameters for SAM."""
    W_base, H_base, angle_rad = sigma_to_rbox_params(sigma_j)

    scale_factor_from_sigma = math.sqrt(sigma_scale)

    final_scale_factor = scale_factor_from_sigma * ellipse_scale_factor

    semi_axis_a = (W_base / 2) * final_scale_factor
    semi_axis_b = (H_base / 2) * final_scale_factor

    cos_theta = torch.cos(angle_rad)
    sin_theta = torch.sin(angle_rad)

    half_width_bbox = torch.sqrt((semi_axis_a * cos_theta)**2 + (semi_axis_b * sin_theta)**2)
    half_height_bbox = torch.sqrt((semi_axis_a * sin_theta)**2 + (semi_axis_b * cos_theta)**2)

    mu_x, mu_y = mu_j[0], mu_j[1]
    x_min = mu_x - half_width_bbox
    y_min = mu_y - half_height_bbox
    x_max = mu_x + half_width_bbox
    y_max = mu_y + half_height_bbox

    bbox_prompt = torch.stack([x_min, y_min, x_max, y_max], dim=-1).detach().cpu().numpy()

    return bbox_prompt.reshape(1, 4)


def segment_anything(image, mu, sigma, device=None, sam_checkpoint=None, model_type=None,
                     label=None, debug=False, mask_filter_config=None, sam_batch_size=8, sam_sample_rules=None):
    """SAM-based mask generation branch with batch loss computation.

    Args:
        image: Input image tensor
        mu: Instance centers
        sigma: Covariance matrices
        device: Computation device
        sam_checkpoint: Path to SAM checkpoint
        model_type: SAM model type
        label: Class labels
        debug: Debug mode flag
        mask_filter_config: Mask filtering configuration
        sam_batch_size: Batch size for loss computation
        sam_sample_rules: Negative sample filtering rules

    Returns:
        loss: Computed loss
        markers: Generated masks
    """
    if debug:
        print("Entering SAM branch:")
    try:
        from mobile_sam import sam_model_registry, SamPredictor
        import numpy as np
        import os
        import time
        from PIL import Image
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Please install MobileSAM: pip install git+https://github.com/ChaoningZhang/MobileSAM.git")

    if device is None:
        device = "cuda"

    img_np = (image - image.min()) / (image.max() - image.min()) * 255.0
    img_np = img_np.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)

    H, W = img_np.shape[:2]
    J = len(mu)

    if sam_checkpoint is None:
        import os
        import time
        from PIL import Image
        from scipy import ndimage
        common_paths = [
            "./mobile_sam.pt"
        ]
        for path in common_paths:
            if os.path.exists(path):
                sam_checkpoint = path
                break
        if sam_checkpoint is None:
            raise ValueError("未找到MobileSAM检查点，请指定sam_checkpoint参数")

    # 初始化/缓存 sam 和 predictor（单例模式，支持 checkpoint/model_type/device 切换）
    if (not hasattr(segment_anything, "sam_model")
        or not hasattr(segment_anything, "model_type")
        or not hasattr(segment_anything, "sam_checkpoint")
        or not hasattr(segment_anything, "device")
        or segment_anything.model_type != model_type
        or segment_anything.sam_checkpoint != sam_checkpoint
        or segment_anything.device != str(device)):
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
        segment_anything.sam_model = sam
        segment_anything.model_type = model_type
        segment_anything.sam_checkpoint = sam_checkpoint
        segment_anything.device = str(device)
        segment_anything.predictor = SamPredictor(sam)  # 同时缓存 predictor
    else:
        sam = segment_anything.sam_model

    predictor = segment_anything.predictor
    # 步骤1: set_image 一次，缓存 image embedding（每张图都必须调用）
    predictor.set_image(img_np)

    points = mu.detach().cpu().numpy()  # (J, 2) 格式: (x, y)
    markers = torch.full((H, W), J+1, dtype=torch.int32, device=mu.device)

    L, V = torch.linalg.eigh(sigma)

    # 步骤2: 准备所有实例的负样本信息
    all_negative_indices = []  # 每个实例的负样本索引列表
    for j in range(J):
        negative_indices = []
        for k in range(J):
            if k != j:
                if sam_sample_rules is not None:
                    skip = False
                    j_label = label[j].item()
                    k_label = label[k].item()
                    dist = np.sqrt(((points[j] - points[k]) ** 2).sum())
                    for filter_pair in sam_sample_rules["filter_pairs"]:
                        class_id1, class_id2, dist_thr = filter_pair
                        if ((j_label == class_id1 and k_label == class_id2) or (j_label == class_id2 and k_label == class_id1)) \
                            and dist < dist_thr:
                            skip = True
                            break
                    if skip:
                        continue
                negative_indices.append(k)
        all_negative_indices.append(negative_indices)

    # 步骤3: 批量 SAM 推理（按负点数量分桶，避免 padding 污染）
    all_masks = [None] * J  # 按原始索引 j 保存 mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # 按负点数量分组（避免 padding）
    from collections import defaultdict
    buckets = defaultdict(list)
    for j in range(J):
        num_neg = len(all_negative_indices[j])
        buckets[num_neg].append(j)

    # 对每个桶进行批量推理
    for num_neg, instance_indices in buckets.items():
        num_points = 1 + num_neg  # 1个正点 + num_neg 个负点

        for batch_start in range(0, len(instance_indices), sam_batch_size):
            batch_end = min(batch_start + sam_batch_size, len(instance_indices))
            batch_size = batch_end - batch_start

            if debug:
                print(f"SAM batch inference: bucket num_neg={num_neg}, processing instances {batch_start} to {batch_end-1}")

            # 构造批量的 point_coords 和 point_labels（无需 padding）
            batch_point_coords = np.zeros((batch_size, num_points, 2), dtype=np.float32)
            batch_point_labels = np.zeros((batch_size, num_points), dtype=np.int32)

            for idx in range(batch_size):
                j = instance_indices[batch_start + idx]

                # 正样本点
                batch_point_coords[idx, 0, :] = points[j]
                batch_point_labels[idx, 0] = 1

                # 负样本点
                neg_indices = all_negative_indices[j]
                for k_idx, k in enumerate(neg_indices):
                    batch_point_coords[idx, k_idx + 1, :] = points[k]
                    batch_point_labels[idx, k_idx + 1] = 0

            # 转换坐标到 SAM 内部坐标系
            coords_transformed = predictor.transform.apply_coords(
                batch_point_coords.reshape(-1, 2), img_np.shape[:2]
            ).reshape(batch_point_coords.shape)

            coords_t = torch.as_tensor(coords_transformed, device=device, dtype=torch.float32)
            labels_t = torch.as_tensor(batch_point_labels, device=device, dtype=torch.int64)

            # 批量推理
            with torch.no_grad():
                image_embedding = predictor.get_image_embedding()  # (1, C, H', W')
                image_pe = sam.prompt_encoder.get_dense_pe()  # (1, C, H', W')

                # Prompt encoder
                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=(coords_t, labels_t),
                    boxes=None,
                    masks=None,
                )

                # 检查维度：image_embeddings/image_pe 必须保持 batch=1
                # mask_decoder 内部会自动 repeat_interleave，所以调用方不能 expand
                B = coords_t.shape[0]
                assert image_embedding.shape[0] == 1, \
                    f"image_embedding must be batch=1, got {image_embedding.shape[0]}"
                assert image_pe.shape[0] == 1, \
                    f"image_pe must be batch=1, got {image_pe.shape[0]}"
                assert sparse_embeddings.shape[0] == B, \
                    f"sparse_embeddings batch mismatch: {sparse_embeddings.shape[0]} vs {B}"
                assert dense_embeddings.shape[0] == B, \
                    f"dense_embeddings batch mismatch: {dense_embeddings.shape[0]} vs {B}"

                # Mask decoder (multimask_output=True 输出3个候选mask)
                # 注意：image_embeddings/image_pe 保持 batch=1，内部会 repeat_interleave
                low_res_masks, iou_preds = sam.mask_decoder(
                    image_embeddings=image_embedding,  # batch=1, 内部会 repeat_interleave
                    image_pe=image_pe,                 # batch=1, 内部会 repeat_interleave
                    sparse_prompt_embeddings=sparse_embeddings,  # batch=B
                    dense_prompt_embeddings=dense_embeddings,    # batch=B
                    multimask_output=True,
                )

                # 上采样回原图大小
                masks_batch = predictor.model.postprocess_masks(
                    low_res_masks,
                    input_size=predictor.input_size,
                    original_size=predictor.original_size,
                )  # (batch_size, 3, H, W)

            masks_batch = masks_batch.cpu().numpy()
            iou_preds = iou_preds.cpu().numpy()  # (batch_size, 3)

            # 处理当前批次的每个实例（需要映射回原始索引）
            for idx in range(batch_size):
                j = instance_indices[batch_start + idx]
                masks = masks_batch[idx]  # (3, H, W)
                scores = iou_preds[idx]   # (3,)

                # 后处理：形态学操作 + 连通域选择
                masks_processed = []
                for mask in masks:
                    mask_uint8 = (mask > 0).astype(np.uint8)
                    mask_opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)

                    num_labels, labels_conn, stats, centroids = cv2.connectedComponentsWithStats(mask_opened)

                    if num_labels > 1:
                        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                        largest_mask = (labels_conn == largest_label)
                        masks_processed.append(largest_mask)
                    else:
                        masks_processed.append(mask_opened > 0)

                masks_processed = np.array(masks_processed)  # (3, H, W)
                best_mask_idx = np.argmax(scores)

                # 掩码筛选
                class_id = label[j].item()
                best_mask_idx, metrics_values, shape_metrics = filter_masks(
                    image, masks_processed, scores, class_id, img_np,
                    points[j], mask_filter_config, debug
                )

                if debug:
                    save_debug_visualization(image, masks_processed, scores, shape_metrics,
                                            metrics_values, best_mask_idx, class_id,
                                            "Optimized Mask Selection")

                # 按原始索引 j 保存 mask（避免 bucket 导致的顺序错乱）
                all_masks[j] = masks_processed[best_mask_idx]

    # 步骤3: 处理掩码并计算损失
    L_pred_list = []
    L_target_list = []

    for j in range(J):
        mask = all_masks[j]
        mask_tensor = torch.from_numpy(mask).to(mu.device)
        markers[mask_tensor] = j + 1

        xy = mask_tensor.nonzero()[:, (1, 0)].float()

        if len(xy) > 0:
            xy_centered = xy - mu[j]
            xy_rotated = V[j].T.matmul(xy_centered[:, :, None])[:, :, 0]

            max_x = torch.max(torch.abs(xy_rotated[:, 0]))
            max_y = torch.max(torch.abs(xy_rotated[:, 1]))
            L_target = torch.stack((max_x, max_y)) ** 2

            L_pred_list.append(L[j])
            L_target_list.append(L_target)

            if debug:
                L_diag = torch.diag_embed(L[j])
                L_target_diag = torch.diag_embed(L_target)
                instance_loss = gwd_sigma_loss(L_diag.unsqueeze(0), L_target_diag.unsqueeze(0).detach())

                visualize_loss_calculation(
                    image, mask_tensor, mu[j], V[j],
                    xy_centered, xy_rotated, max_x, max_y,
                    L[j], L_target, instance_loss,
                    j,label[j].item() 
                )

    # 步骤4: 批量计算损失（简单平均）
    if len(L_pred_list) > 0:
        L_pred_batch = torch.stack([torch.diag_embed(l) for l in L_pred_list])
        L_target_batch = torch.stack([torch.diag_embed(l) for l in L_target_list])

        loss = gwd_sigma_loss(L_pred_batch, L_target_batch.detach(), reduction='none')
        final_loss = loss.mean()
    else:
        final_loss = mu.new_tensor(0.0)

    return final_loss, markers


def voronoi_watershed_loss(mu, sigma, label, image, pos_thres=0.994, neg_thres=0.005,
                          down_sample=2, topk=0.95, default_sigma=4096,
                          voronoi='gaussian-orientation', alpha=0.1, debug=False):
    """Voronoi+Watershed based mask generation branch.

    Args:
        mu: Instance centers
        sigma: Covariance matrices
        label: Class labels
        image: Input image
        pos_thres: Positive threshold for Voronoi
        neg_thres: Negative threshold for Voronoi
        down_sample: Downsample factor
        topk: Top-k ratio for loss calculation
        default_sigma: Default sigma value
        voronoi: Voronoi type ('standard', 'gaussian-orientation', 'gaussian-full')
        alpha: Alpha parameter
        debug: Debug mode flag

    Returns:
        loss: Computed loss
        (vor, markers): Voronoi map and watershed markers
    """
    J = len(sigma)
    if J == 0:
        return sigma.sum()
    D = down_sample
    H, W = image.shape[-2:]
    if debug:
        print(f'Gaussian Voronoi Watershed Loss: {H}x{W}, downsample={D}, J={J}')
        print(f'default_sigma={default_sigma}, voronoi={voronoi}, alpha={alpha}')
    h, w = H // D, W // D
    x = torch.linspace(0, h, h, device=mu.device)
    y = torch.linspace(0, w, w, device=mu.device)
    xy = torch.stack(torch.meshgrid(x, y, indexing='xy'), -1)
    vor = mu.new_zeros(J, h, w)
    # Get distribution for each instance
    mm = (mu.detach() / D).round()

    if voronoi == 'standard':
        sg = sigma.new_tensor((default_sigma, 0, 0, default_sigma)).reshape(2, 2)
        sg = sg / D ** 2
        for j, m in enumerate(mm):
            vor[j] = gaussian_2d(xy.view(-1, 2), m[None], sg[None]).view(h, w)
    elif voronoi == 'gaussian-orientation':
        L, V = torch.linalg.eigh(sigma)
        L = L.detach().clone()
        L = L / (L[:, 0:1] * L[:, 1:2]).sqrt() * default_sigma
        sg = V.matmul(torch.diag_embed(L)).matmul(V.permute(0, 2, 1)).detach()
        sg = sg / D ** 2
        for j, (m, s) in enumerate(zip(mm, sg)):
            vor[j] = gaussian_2d(xy.view(-1, 2), m[None], s[None]).view(h, w)
    elif voronoi == 'gaussian-full':
        sg = sigma.detach() / D ** 2
        for j, (m, s) in enumerate(zip(mm, sg)):
            vor[j] = gaussian_2d(xy.view(-1, 2), m[None], s[None]).view(h, w)
    # val: max prob, vor: belong to which instance, cls: belong to which class
    val, vor = torch.max(vor, 0)
    if D > 1:
        vor = vor[:, None, :, None].expand(-1, D, -1, D).reshape(H, W)
        val = torch.nn.functional.interpolate(
            val[None, None], (H, W), mode='bilinear', align_corners=True)[0, 0]
    cls = label[vor]
    kernel = val.new_ones((1, 1, 3, 3))
    kernel[0, 0, 1, 1] = -8
    ridges = torch.conv2d(vor[None].float(), kernel, padding=1)[0] != 0
    vor += 1
    pos_thres = val.new_tensor(pos_thres)
    neg_thres = val.new_tensor(neg_thres)
    vor[val < pos_thres[cls]] = 0
    vor[val < neg_thres[cls]] = J + 1
    vor[ridges] = J + 1

    cls_bg = torch.where(vor == J + 1, 16, cls)
    cls_bg = torch.where(vor == 0, -1, cls_bg)

    # PyTorch does not support watershed, use cv2
    img_uint8 = (image - image.min()) / (image.max() - image.min()) * 255
    img_uint8 = img_uint8.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    img_uint8 = cv2.medianBlur(img_uint8, 3)
    markers = vor.detach().cpu().numpy().astype(np.int32)
    markers = vor.new_tensor(cv2.watershed(img_uint8, markers))
    if debug:
        plot_gaussian_voronoi_watershed(image, cls_bg, markers, labels=label)

    L, V = torch.linalg.eigh(sigma)
    L_target = []
    for j in range(J):
        xy = (markers == j + 1).nonzero()[:, (1, 0)].float()
        if len(xy) == 0:
            L_target.append(L[j].detach())
            continue
        xy = xy - mu[j]
        xy = V[j].T.matmul(xy[:, :, None])[:, :, 0]
        max_x = torch.max(torch.abs(xy[:, 0]))
        max_y = torch.max(torch.abs(xy[:, 1]))
        L_target.append(torch.stack((max_x, max_y)) ** 2)
    L_target = torch.stack(L_target)
    L = torch.diag_embed(L)
    L_target = torch.diag_embed(L_target)
    loss = gwd_sigma_loss(L, L_target.detach(), reduction='none')  # 必须是 'none' 才能配合 topk
    loss = torch.topk(loss, int(np.ceil(len(loss) * topk)), largest=False)[0].mean()
    return loss, (vor, markers)


def get_loss_from_mask(mu, sigma, label, image, pos_thres, neg_thres, down_sample=2,
                      topk=0.95, default_sigma=4096, voronoi='gaussian-orientation',
                      alpha=0.1, debug=False, mask_filter_config=None,
                      sam_checkpoint='./mobile_sam.pt', model_type='vit_t',
                      sam_instance_thr=-1, sam_batch_size=8, device=None, sam_sample_rules=None):
    """Main entry point for mask-based loss calculation.

    Switches between SAM and Voronoi+Watershed based on instance count.

    Args:
        mu: Instance centers
        sigma: Covariance matrices
        label: Class labels
        image: Input image
        pos_thres: Positive threshold
        neg_thres: Negative threshold
        down_sample: Downsample factor
        topk: Top-k ratio
        default_sigma: Default sigma value
        voronoi: Voronoi type
        alpha: Alpha parameter
        debug: Debug mode flag
        mask_filter_config: Mask filtering configuration
        sam_checkpoint: SAM checkpoint path
        model_type: SAM model type
        sam_instance_thr: Threshold to switch to SAM branch
        sam_batch_size: Batch size for SAM inference
        device: Computation device
        sam_sample_rules: Negative sample filtering rules

    Returns:
        loss: Computed loss
        (vor, markers): Generated masks
    """
    if debug:
        print(f"SAM config: checkpoint={sam_checkpoint}, model_type={model_type}, sam_instance_thr={sam_instance_thr}, sam_batch_size={sam_batch_size}")
    J = len(sigma)
    if J == 0:
        return sigma.sum()
    if J <= sam_instance_thr:
        loss, markers = segment_anything(
            image, mu, sigma,
            device=mu.device,
            sam_checkpoint=sam_checkpoint,
            model_type=model_type,
            label=label,
            debug=debug,
            mask_filter_config=mask_filter_config,
            sam_batch_size=sam_batch_size,
            sam_sample_rules=sam_sample_rules,
        )
        vor = markers.clone()
        return loss, (vor, markers)
    else:
        loss, (vor, markers) = voronoi_watershed_loss(
            mu, sigma, label, image,
            pos_thres, neg_thres, down_sample, topk,
            default_sigma, voronoi, alpha,
            debug=debug,
        )
        return loss, (vor, markers)


@MODELS.register_module()
class PGDMLoss(nn.Module):
    """Pseudo Ground Truth Distribution Mask Loss.

    Originally known as VoronoiWatershedLoss. Generates pseudo ground truth masks
    from point annotations using either SAM or Voronoi+Watershed approach.

    Args:
        loss_weight (float): Loss weight. Defaults to 1.0.
        down_sample (int): Downsample factor for Voronoi. Defaults to 2.
        topk (float): Top-k ratio for loss calculation. Defaults to 0.95.
        alpha (float): Alpha parameter. Defaults to 0.1.
        default_sigma (int): Default sigma value for Voronoi. Defaults to 4096.
        debug (bool): Debug mode flag. Defaults to False.
        mask_filter_config (dict): Mask filtering configuration. Defaults to None.
        sam_instance_thr (int): Instance count threshold to switch to SAM. Defaults to -1 (disabled).
        sam_batch_size (int): Batch size for SAM inference. Defaults to 8.
        sam_sample_rules (dict): Negative sample filtering rules. Defaults to None.
        use_class_specific_watershed (bool): Use class-specific watershed. Defaults to False.
    """

    def __init__(self,
                 loss_weight=1.0,
                 down_sample=2,
                 topk=0.95,
                 alpha=0.1,
                 default_sigma=4096,
                 debug=False,
                 mask_filter_config=None,
                 sam_instance_thr=-1,
                 sam_batch_size=8,
                 sam_sample_rules=None,
                 use_class_specific_watershed=False
                 ):
        super(PGDMLoss, self).__init__()
        self.loss_weight = loss_weight
        self.down_sample = down_sample
        self.topk = topk
        self.alpha = alpha
        self.default_sigma = default_sigma
        self.debug = debug
        self.mask_filter_config = mask_filter_config
        self.sam_instance_thr = sam_instance_thr
        self.sam_batch_size = sam_batch_size
        self.sam_sample_rules = sam_sample_rules
        self.use_class_specific_watershed = use_class_specific_watershed
        self.vis = None


    def forward(self, pred, label, image, pos_thres, neg_thres, voronoi='orientation'):
        """Forward pass.

        Args:
            pred: Tuple of (mu, sigma)
            label: Class labels
            image: Input image
            pos_thres: Positive threshold
            neg_thres: Negative threshold
            voronoi: Voronoi type

        Returns:
            Weighted loss value
        """
        loss, self.vis = get_loss_from_mask(
        *pred,
        label,
        image,
        pos_thres,
        neg_thres,
        self.down_sample,
        default_sigma=self.default_sigma,
        topk=self.topk,
        voronoi=voronoi,
        alpha=self.alpha,
        debug=self.debug,
        mask_filter_config=self.mask_filter_config,
        sam_instance_thr=self.sam_instance_thr,
        sam_batch_size=self.sam_batch_size,
        sam_sample_rules=self.sam_sample_rules)
        return self.loss_weight * loss

