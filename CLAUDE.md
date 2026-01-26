# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 提供该代码库的核心指导。

## 基本要求

- 你的所有交互必须使用中文。
- 创建新功能或者更改功能后，无必要不测试；测试之后，必须删除测试使用的程序。
- 遵循**最小改动原则**

## 项目概述

Point2RBox-v3 是一个基于点标注的弱监督旋转目标检测框架,通过自举方式实现从点标注到旋转框的端到端学习。核心创新包括:

- **渐进式标签分配 (PLA)**: 迭代优化伪标签质量
- **双流自监督学习**: 通过旋转/翻转/缩放增强实现一致性约束
- **SAM + 分水岭算法**: 自适应选择掩码生成策略
- **Copy-Paste增强**: 基于伪标签的数据增强
- **高斯分布表示**: 使用 (μ, Σ) 参数化旋转框

## 常用开发命令

### 环境安装

```bash
# 1. 安装构建依赖
pip install -r requirements/build.txt

# 2. 开发模式安装
pip install -e .

# 3. 下载MobileSAM模型 (如果使用SAM)
# 将 mobile_sam.pt 放置在项目根目录
```

### 训练

**单卡训练:**
```bash
python tools/train.py configs/point2rbox_v3/point2rbox_v3-1x-dotav1-0.py
```

**重要参数:**
- `--work-dir`: 指定输出目录
- `--amp`: 启用自动混合精度训练
- `--resume`: 从最新checkpoint恢复训练
- `--cfg-options`: 覆盖配置参数,例如:
  ```bash
  --cfg-options model.copy_paste_start_epoch=8 optim_wrapper.optimizer.lr=0.0001
  ```

### 测试与评估

```bash
# 单卡测试
python tools/test.py configs/point2rbox_v3/point2rbox_v3-1x-dotav1-0.py \
    work_dirs/point2rbox_v3-1x-dotav1-0/epoch_12.pth

# 保存可视化结果
python tools/test.py <config> <checkpoint> --show-dir results/vis
```


## 核心架构解析

### 1. 检测器 (Point2RBoxV3)

**文件**: `mmrotate/models/detectors/point2rbox_v3.py`

**关键流程**:

```python
# 训练时的前向流程:
1. prepare_dual_stream_inputs()  # 生成原始图像 + 增强图像的双流输入
   - 随机选择增强方式: 旋转 (ss_prob[0]) / 翻转 (ss_prob[1]) / 缩放 (ss_prob[2])
   - 为增强样本设置 bids[:, 2] = 1 标记

2. prepare_edges()  # 使用TED模型提取边缘特征 (epoch >= edge_loss_start_epoch)

3. prepare_copy_paste_step2()  # Copy-Paste增强 (epoch >= copy_paste_start_epoch)
   - 使用上一轮缓存的模式进行粘贴

4. generate_pseudo_targets() / bbox_head.predict()  # 生成伪标签引导金字塔分配（核心）
   - epoch < label_assign_pseudo_label_switch_eopch: 使用Voronoi分水岭作为伪标签
   - epoch >= label_assign_pseudo_label_switch_eopch: 使用上一轮模型预测的结果作为伪标签

5. bbox_head.loss()  # 计算多个损失
```

**核心参数**:
- `ss_prob=[0.68, 0.07, 0.25]`: 自监督采样概率 [旋转, 翻转, 缩放]
- `copy_paste_start_epoch=6`: Copy-Paste开始轮次
- `label_assign_pseudo_label_switch_eopch=6`: 切换标签分配策略的轮次
- `rotate_range=(0.25, 0.75)`: 旋转角度范围 (π的倍数)
- `scale_range=(0.5, 0.9)`: 缩放范围

**双流机制**:
- 原始流: batch_id = 0~N-1, bids[:, 2] = 0
- 增强流: batch_id = N~2N-1, bids[:, 2] = 1
- 通过 `bids` 字段追踪: `[batch_id, is_synthetic, is_augmented, object_id]`

### 2. 检测头 (Point2RBoxV3Head)

**文件**: `mmrotate/models/dense_heads/point2rbox_v3_head.py`

**高斯表示**:
```python
# 预测输出:
sig_x, sig_y = bbox_pred[:, 0].exp(), bbox_pred[:, 1].exp()  # 高斯标准差
dx, dy = bbox_pred[:, 2:4].sigmoid() * 2 - 1  # 中心偏移 (-1, 1)
angle_pred  # PSC角度编码 (3通道)

# 转换为旋转框:
Σ = R @ diag(sig_x, sig_y) @ R^T  # 协方差矩阵
μ = grid_center + (dx, dy) * stride  # 中心坐标
```

**关键参数**:
- `voronoi_type`: Voronoi图类型
  - `'standard'`: 使用固定方差的高斯
  - `'gaussian-orientation'`: 使用预测角度的各向异性高斯
  - `'gaussian-full'`: 使用完整预测的协方差矩阵
- `voronoi_thres`: 类别特定阈值,格式:
  ```python
  dict(
      default=[0.994, 0.005],  # [pos_thres, neg_thres]
      override=(
          ([2, 11], [0.999, 0.6]),     # 船舶、直升机使用更严格阈值
          ([7, 8, 10, 14], [0.95, 0.005])  # 部分类别使用更宽松阈值
      )
  )
  ```
- `square_cls=[1, 9, 11]`: 正方形类别(棒球场、网球场、直升机),角度不敏感
- `edge_loss_cls`: 应用边缘损失的类别列表
- `post_process={11: 1.2}`: 后处理缩放因子(类别ID -> 缩放倍数)

### 3. 损失函数体系

**注意**: 从v3版本开始，损失函数模块已重构：
- `point2rbox_v2_loss.py` → `point2rbox_v3_loss.py`
- `PGDMLoss` → `PGDMLoss` (解耦到 `pgdm_loss.py`)

#### PGDMLoss (伪地面真值分布掩码损失，核心)

**别名**: `PGDMLoss` (保留向后兼容性)
**文件**: `mmrotate/models/losses/pgdm_loss.py`

**作用**: 从点标注生成高质量掩码作为伪旋转框标注

**两种模式**:

1. **SAM模式** (当 `J <= sam_instance_thr` 时): 使用MobileSAM生成掩码
   - 计算掩码得分、掩码筛选：`mmrotate/models/losses/utils.py`
   - 正样本点: 当前实例中心
   - 负样本点: 其他实例中心 (根据sam_sample_rules过滤)
   - 掩码后处理: 形态学开运算 + 连通域选择
   - 掩码筛选: 基于shape metrics (圆度、矩形度、颜色一致性、中心对齐度等)

2. **Voronoi+分水岭模式** (当 `J > sam_instance_thr` 时)：使用Watershed生成掩码
   ```python
   # 计算Voronoi图
   for j in range(J):
       vor[j] = Gaussian_2D(xy, μ_j, Σ_j)  # 根据voronoi_type选择Σ

   # 阈值过滤
   vor[val < pos_thres[cls]] = 0      # 不确定区域
   vor[val < neg_thres[cls]] = J+1    # 背景
   vor[ridges] = J+1                   # Voronoi边界

   # 分水岭算法
   markers = cv2.watershed(image, vor)

   ```

   # 计算损失
   把mask绕着annotation，按照高斯分布给出的方向信息，旋转到和预测框一个方向，取其`max(+x,-x)`作为伪标签的宽，`max(+y,-y)`作为伪标签的高，然后和预测框比较计算损失。

**配置参数**:
- `sam_instance_thr`: SAM模式阈值,建议值: 4
  - 设为-1禁用SAM
- `mask_filter_config`: SAM掩码筛选配置,示例:
  ```python
  {
      'default': {
          'required_metrics': ['color_consistency', 'center_alignment'],
          'weights': {'color_consistency': 6, 'center_alignment': 10}
      },
      2: {  # 船舶类别
          'required_metrics': ['rectangularity', 'aspect_ratio_reasonableness'],
          'weights': {'rectangularity': 8, 'aspect_ratio_reasonableness': 5},
          'aspect_ratio_range': (1.5, 8.0)
      }
  }
  ```
- `sam_sample_rules`: 负样本过滤规则:
  ```python
  {
      'filter_pairs': [
          (1, 2, 100),  # 飞机和船舶距离<100像素时不作为负样本
          (3, 4, 50)
      ]
  }
  ```
- `sam_batch_size`: SAM批量推理的batch size,避免OOM (建议值: 4)

#### SAM批量推理机制

**文件**: `mmrotate/models/losses/pgdm_loss.py` (lines 165-333)

**目的**: 对同一张图像中的多个实例进行批量SAM推理,避免重复计算image embedding

**核心原理**:

1. **单例模式缓存** (避免重复加载模型):
   ```python
   # 缓存 sam_model 和 predictor,支持参数变更检测
   if (not hasattr(segment_anything, "sam_model")
       or segment_anything.sam_checkpoint != sam_checkpoint
       or segment_anything.device != str(device)):
       sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
       segment_anything.sam_model = sam
       segment_anything.predictor = SamPredictor(sam)  # 同时缓存 predictor

   predictor = segment_anything.predictor
   predictor.set_image(img_np)  # 每张图必须调用一次,缓存 image embedding
   ```

2. **Bucket-based Batching** (避免padding污染):
   ```python
   # 按负点数量分组,避免 np.zeros padding 产生 (0,0) 假负点
   buckets = defaultdict(list)
   for j in range(J):
       num_neg = len(all_negative_indices[j])
       buckets[num_neg].append(j)  # 同一组内负点数量相同,无需 padding

   # 对每个桶分别进行批量推理
   for num_neg, instance_indices in buckets.items():
       for batch_start in range(0, len(instance_indices), sam_batch_size):
           # 构造批量输入: coords_t (B, N, 2), labels_t (B, N)
           ...
   ```

3. **关键维度约束** (**必须遵守,否则维度爆炸**):
   ```python
   # MobileSAM 的 mask_decoder.predict_masks() 内部会对 image_embeddings
   # 执行 repeat_interleave(..., tokens.shape[0]),因此调用方必须传入 batch=1

   B = coords_t.shape[0]  # 批次大小

   # 严格检查: image_embeddings/image_pe 必须是 batch=1
   assert image_embedding.shape[0] == 1, "必须保持 batch=1"
   assert image_pe.shape[0] == 1, "必须保持 batch=1"

   # sparse/dense embeddings 才是真正的 batch=B
   assert sparse_embeddings.shape[0] == B
   assert dense_embeddings.shape[0] == B

   # 调用 mask_decoder (不要对 image_embedding 做 expand!)
   low_res_masks, iou_preds = sam.mask_decoder(
       image_embeddings=image_embedding,  # (1, C, H', W'), 内部会 repeat
       image_pe=image_pe,                 # (1, C, H', W'), 内部会 repeat
       sparse_prompt_embeddings=sparse_embeddings,  # (B, N, C)
       dense_prompt_embeddings=dense_embeddings,    # (B, C, H', W')
       multimask_output=True,
   )
   # 输出: low_res_masks (B, 3, H', W'), iou_preds (B, 3)
   ```


#### 其他损失函数

省略

### 4. 调试与可视化工具

**文件**: `mmrotate/models/losses/vis.py`

#### 开启调试模式

在配置文件中设置:
```python
model = dict(
    type='Point2RBoxV3',
    debug=True,  # 开启检测器调试
    bbox_head=dict(
        loss_pgdm=dict(
            type='PGDMLoss',  # 或 'PGDMLoss' (向后兼容)
            debug=True  # 开启损失函数调试
        )
    )
)
```

#### 生成的可视化文件

1. **debug/** 目录:
   - `{img_id}_{i}.png`: 预测框可视化
   - `{timestamp}-Gaussian-Voronoi-2x3.png`: Voronoi分水岭过程(6子图布局)

2. **vis_loss/** 目录:
   - `loss_calc_{timestamp}_{class}_{inst}.png`: 损失计算详细过程(6子图):
     - 原始图像 + 中心点
     - SAM掩码叠加
     - 中心化坐标
     - 旋转后坐标 + 最大范围
     - 特征值对比表格
     - 特征向量与椭圆可视化

3. **vis/** 目录:
   - `{timestamp}-{class}-masks.png`: SAM多掩码评分对比
   - Copy-Paste模式可视化

#### 形状指标计算

```python
from mmrotate.models.losses.utils import calculate_shape_metrics

metrics = calculate_shape_metrics(
    image=image,  # torch.Tensor
    mask=mask,    # np.ndarray
    required_metrics=['circularity', 'rectangularity', 'color_consistency',
                     'aspect_ratio_reasonableness', 'center_alignment'],
    original_image=img_np,
    aspect_ratio_range=(1.0, 10.0),
    prompt_point=point,
    debug=True
)
# 返回: {'circularity': 0.85, 'rectangularity': 0.92, ...}
```

**指标说明**:
- `circularity`: 圆度 = 轮廓面积 / 最小外接圆面积
- `rectangularity`: 矩形度 = 轮廓面积 / 最小旋转矩形面积
- `color_consistency`: 颜色一致性 = exp(-加权标准差/30)
- `aspect_ratio_reasonableness`: 长宽比合理性
- `center_alignment`: 中心对齐度 (提示点是否在框内)

