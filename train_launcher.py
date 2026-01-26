#!/usr/bin/env python3
"""
交互式训练启动脚本 - Point2RBox-v3
自动检测GPU、选择配置文件、设置训练参数
"""

import subprocess
import re
import sys
from pathlib import Path


def get_gpu_info():
    """获取GPU信息，返回[(gpu_id, memory_free, memory_total), ...]"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(',')
                gpu_id = int(parts[0].strip())
                mem_free = int(parts[1].strip())
                mem_total = int(parts[2].strip())
                gpus.append((gpu_id, mem_free, mem_total))
        return gpus
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("警告: 无法获取GPU信息，nvidia-smi不可用")
        return []


def display_gpus(gpus):
    """显示GPU信息表格"""
    if not gpus:
        print("未检测到GPU")
        return

    print("\n" + "="*70)
    print(f"{'GPU ID':<8} {'显存空闲':<15} {'显存总量':<15} {'使用率':<10}")
    print("="*70)

    for gpu_id, mem_free, mem_total in gpus:
        mem_used = mem_total - mem_free
        usage_pct = (mem_used / mem_total * 100) if mem_total > 0 else 0
        print(f"{gpu_id:<8} {mem_free:<15} {mem_total:<15} {usage_pct:>6.1f}%")

    print("="*70)


def select_gpu(gpus):
    """选择GPU，返回GPU ID"""
    if not gpus:
        return None

    display_gpus(gpus)

    # 自动选择显存最空闲的卡
    best_gpu = max(gpus, key=lambda x: x[1])
    best_gpu_id = best_gpu[0]

    print(f"\n推荐GPU: {best_gpu_id} (空闲显存: {best_gpu[1]} MB)")
    user_input = input(f"选择GPU ID (直接回车使用推荐): ").strip()

    if not user_input:
        return best_gpu_id

    try:
        selected_id = int(user_input)
        if any(gpu[0] == selected_id for gpu in gpus):
            return selected_id
        else:
            print(f"警告: GPU {selected_id} 不存在，使用推荐GPU {best_gpu_id}")
            return best_gpu_id
    except ValueError:
        print(f"输入无效，使用推荐GPU {best_gpu_id}")
        return best_gpu_id


def select_config():
    """选择配置文件"""
    configs = {
        '1': ('configs/point2rbox_v3/point2rbox_v3-1x-dotav1-0.py', 'DOTA v1.0 训练'),
        '2': ('configs/point2rbox_v3/point2rbox_v3-1x-dotav1-5.py', 'DOTA v1.5 训练'),
        '3': ('configs/point2rbox_v3/point2rbox_v3-1x-dior.py', 'DIOR 训练'),
        '4': ('configs/point2rbox_v3/point2rbox_v3-1x-star.py', 'STAR 训练'),
        '5': ('configs/point2rbox_v3/point2rbox_v3-pseudo-generator-dotav1-0.py', 'DOTA v1.0 伪标签生成'),
        '6': ('configs/point2rbox_v3/point2rbox_v3-pseudo-generator-dior.py', 'DIOR 伪标签生成'),
        '7': ('configs/point2rbox_v3/point2rbox_v3-pseudo-generator-star.py', 'STAR 伪标签生成'),
    }

    print("\n" + "="*70)
    print("可用配置文件:")
    print("="*70)
    for key, (path, desc) in configs.items():
        print(f"  {key}. {desc}")
        print(f"     {path}")
    print("="*70)

    default_choice = '1'
    user_input = input(f"\n选择配置 (直接回车使用 DOTA v1.0): ").strip()

    if not user_input:
        user_input = default_choice

    if user_input in configs:
        return configs[user_input][0]
    else:
        print(f"输入无效，使用默认配置 (DOTA v1.0)")
        return configs[default_choice][0]


def get_training_params():
    """获取训练参数"""
    print("\n" + "="*70)
    print("训练参数设置 (直接回车使用默认值)")
    print("="*70)

    params = {}

    # Work directory
    work_dir = input("工作目录 work-dir (默认: 自动生成): ").strip()
    if work_dir:
        params['work_dir'] = work_dir

    # AMP
    amp = input("启用混合精度训练 AMP? (y/n, 默认: n): ").strip().lower()
    params['amp'] = amp == 'y'

    # Resume
    resume = input("从最新checkpoint恢复训练? (y/n, 默认: n): ").strip().lower()
    params['resume'] = resume == 'y'

    # Custom config options
    print("\n配置覆盖 (格式: key1=value1 key2=value2, 例如: model.copy_paste_start_epoch=8)")
    cfg_options = input("cfg-options (直接回车跳过): ").strip()
    if cfg_options:
        params['cfg_options'] = cfg_options

    return params


def build_command(config_path, gpu_id, params):
    """构建训练命令"""
    cmd = []

    # 设置CUDA_VISIBLE_DEVICES
    if gpu_id is not None:
        cmd.append(f"CUDA_VISIBLE_DEVICES={gpu_id}")

    # Python命令
    cmd.append("python tools/train.py")
    cmd.append(config_path)

    # Work directory
    if 'work_dir' in params:
        cmd.append(f"--work-dir {params['work_dir']}")

    # AMP
    if params.get('amp', False):
        cmd.append("--amp")

    # Resume
    if params.get('resume', False):
        cmd.append("--resume")

    # Config options
    if 'cfg_options' in params:
        cmd.append(f"--cfg-options {params['cfg_options']}")

    return ' '.join(cmd)


def main():
    print("="*70)
    print(" " * 20 + "Point2RBox-v3 训练启动器")
    print("="*70)

    # 1. 选择GPU
    gpus = get_gpu_info()
    gpu_id = select_gpu(gpus)

    # 2. 选择配置
    config_path = select_config()

    # 3. 获取训练参数
    params = get_training_params()

    # 4. 构建命令
    command = build_command(config_path, gpu_id, params)

    # 5. 显示最终命令并确认
    print("\n" + "="*70)
    print("即将执行的命令:")
    print("="*70)
    print(command)
    print("="*70)

    confirm = input("\n确认执行? (y/n, 默认: y): ").strip().lower()
    if confirm == 'n':
        print("已取消")
        sys.exit(0)

    print("\n开始训练...\n")

    # 6. 执行命令
    subprocess.run(command, shell=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n训练已中断")
        sys.exit(1)
