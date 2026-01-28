sam_instance_thr = 4 # 认为这个图可以进入SAM的instance数量的阈值
sam_batch_size = 4  # SAM 批量推理的 batch size，避免 OOM
sam_enable_predictor_cache = False  # 是否启用predictor缓存（False避免精度下降）
sam_no_batch_inference = True  # 是否禁用批量推理，使用官方单个推理方式（True避免精度下降）
mask_filter_config=dict(
         {
         'default': {
             'required_metrics': ['color_consistency', 'center_alignment'],
             'weights': {'color_consistency': 2, 'center_alignment': 10}
         }
         }
         )
sam_sample_rules = dict({
    "filter_pairs": [(34, 39, 200)]})