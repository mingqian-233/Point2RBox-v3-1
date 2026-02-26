#!/bin/bash
# Point2RBox-v3 快速测试脚本

set -e
# 确保在仓库根目录运行（以脚本所在目录为准）
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# 让 Python 一定能找到 third_parties 等顶层目录
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}                  Point2RBox-v3 测试启动器${NC}"
echo -e "${BLUE}======================================================================${NC}"

# ============= 激活 Conda 环境 =============
echo ""
echo -e "${GREEN}激活 pr3 环境...${NC}"

# 初始化 conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo -e "${RED}错误: 找不到 conda 初始化脚本${NC}"
    echo -e "${YELLOW}请手动激活 pr3 环境后再运行此脚本${NC}"
    exit 1
fi

# 激活 pr3 环境
conda activate pr3
if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 无法激活 pr3 环境${NC}"
    echo -e "${YELLOW}请确保 pr3 环境已创建${NC}"
    exit 1
fi

echo -e "${GREEN}✓ pr3 环境已激活${NC}"
echo -e "${YELLOW}当前 Python: $(which python)${NC}"
echo ""

# ============= GPU选择 =============
echo ""
echo -e "${GREEN}可用GPU信息:${NC}"
echo "======================================================================"
nvidia-smi --query-gpu=index,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits | \
    awk -F',' '{printf "GPU %s: 显存 %s/%s MB (使用率: %s%%)\n", $1, $3-$2, $3, $4}'
echo "======================================================================"

# 自动选择显存最大的GPU
BEST_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
    sort -t',' -k2 -rn | head -1 | cut -d',' -f1)

echo ""
echo -e "${YELLOW}推荐GPU: ${BEST_GPU}${NC}"
read -p "选择GPU ID (直接回车使用推荐): " GPU_ID
GPU_ID=${GPU_ID:-$BEST_GPU}

# ============= 配置文件选择 =============
echo ""
echo -e "${GREEN}可用配置文件:${NC}"
echo "======================================================================"
echo "  1. DOTA v1.0 测试 (默认)"
echo "     configs/point2rbox_v3/point2rbox_v3-1x-dotav1-0.py"
echo ""
echo "  2. DOTA v1.5 测试"
echo "     configs/point2rbox_v3/point2rbox_v3-1x-dotav1-5.py"
echo ""
echo "  3. DIOR 测试"
echo "     configs/point2rbox_v3/point2rbox_v3-1x-dior.py"
echo ""
echo "  4. STAR 测试"
echo "     configs/point2rbox_v3/point2rbox_v3-1x-star.py"
echo ""
echo "  5. 自定义配置路径"
echo "======================================================================"

read -p "选择配置 (直接回车使用默认): " CONFIG_CHOICE
CONFIG_CHOICE=${CONFIG_CHOICE:-1}

case $CONFIG_CHOICE in
    1)
        CONFIG="configs/point2rbox_v3/point2rbox_v3-1x-dotav1-0.py"
        DEFAULT_WORK_DIR="work_dirs/point2rbox_v3-1x-dotav1-0"
        ;;
    2)
        CONFIG="configs/point2rbox_v3/point2rbox_v3-1x-dotav1-5.py"
        DEFAULT_WORK_DIR="work_dirs/point2rbox_v3-1x-dotav1-5"
        ;;
    3)
        CONFIG="configs/point2rbox_v3/point2rbox_v3-1x-dior.py"
        DEFAULT_WORK_DIR="work_dirs/point2rbox_v3-1x-dior"
        ;;
    4)
        CONFIG="configs/point2rbox_v3/point2rbox_v3-1x-star.py"
        DEFAULT_WORK_DIR="work_dirs/point2rbox_v3-1x-star"
        ;;
    5)
        read -p "输入配置文件路径: " CONFIG
        DEFAULT_WORK_DIR=""
        ;;
    *)
        CONFIG="configs/point2rbox_v3/point2rbox_v3-1x-dotav1-0.py"
        DEFAULT_WORK_DIR="work_dirs/point2rbox_v3-1x-dotav1-0"
        echo -e "${YELLOW}输入无效，使用默认配置${NC}"
        ;;
esac

# ============= Checkpoint选择 =============
echo ""
echo -e "${GREEN}Checkpoint选择:${NC}"
echo "======================================================================"

# 尝试列出工作目录中的checkpoint文件
if [ ! -z "$DEFAULT_WORK_DIR" ] && [ -d "$DEFAULT_WORK_DIR" ]; then
    echo -e "${YELLOW}在 $DEFAULT_WORK_DIR 中找到的checkpoint:${NC}"
    CKPTS=($(find "$DEFAULT_WORK_DIR" -name "*.pth" -type f 2>/dev/null | sort))
    if [ ${#CKPTS[@]} -gt 0 ]; then
        for i in "${!CKPTS[@]}"; do
            echo "  $((i+1)). ${CKPTS[$i]}"
        done
        echo ""
        read -p "选择checkpoint序号或输入完整路径: " CKPT_CHOICE

        # 判断输入是数字还是路径
        if [[ "$CKPT_CHOICE" =~ ^[0-9]+$ ]] && [ "$CKPT_CHOICE" -ge 1 ] && [ "$CKPT_CHOICE" -le ${#CKPTS[@]} ]; then
            CHECKPOINT="${CKPTS[$((CKPT_CHOICE-1))]}"
        else
            CHECKPOINT="$CKPT_CHOICE"
        fi
    else
        echo -e "${YELLOW}未找到checkpoint文件${NC}"
        read -p "输入checkpoint文件路径: " CHECKPOINT
    fi
else
    read -p "输入checkpoint文件路径: " CHECKPOINT
fi

# 验证checkpoint文件存在
if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${RED}错误: Checkpoint文件不存在: $CHECKPOINT${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 选择的checkpoint: $CHECKPOINT${NC}"

# ============= 测试参数 =============
echo ""
echo -e "${GREEN}测试参数设置 (直接回车使用默认值):${NC}"
echo "======================================================================"

# Work directory
read -p "工作目录 (默认: 自动生成): " WORK_DIR

# Show directory
read -p "保存可视化结果目录 (直接回车跳过): " SHOW_DIR

# Output pickle file
read -p "保存预测结果到pickle文件 (直接回车跳过): " OUT_FILE

# Config options
echo ""
echo "配置覆盖 (例如: model.test_cfg.nms.iou_threshold=0.1)"
read -p "cfg-options (直接回车跳过): " CFG_OPTIONS

# ============= 构建命令 =============
CMD="CUDA_VISIBLE_DEVICES=$GPU_ID python tools/test.py $CONFIG $CHECKPOINT"

if [ ! -z "$WORK_DIR" ]; then
    CMD="$CMD --work-dir $WORK_DIR"
fi

if [ ! -z "$SHOW_DIR" ]; then
    CMD="$CMD --show-dir $SHOW_DIR"
fi

if [ ! -z "$OUT_FILE" ]; then
    CMD="$CMD --out $OUT_FILE"
fi

if [ ! -z "$CFG_OPTIONS" ]; then
    CMD="$CMD --cfg-options $CFG_OPTIONS"
fi

# ============= 确认并执行 =============
echo ""
echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN}即将执行的命令:${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo -e "${YELLOW}$CMD${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo ""

read -p "确认执行? (y/n, 默认: y): " CONFIRM
CONFIRM=${CONFIRM:-y}

if [ "$CONFIRM" != "y" ]; then
    echo -e "${RED}已取消${NC}"
    exit 0
fi

echo ""
echo -e "${GREEN}开始测试...${NC}"
echo ""

# 执行测试
eval $CMD
