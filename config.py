# config.py
"""
配置模块
--------------------------
    统一管理训练、数据、模型选择与运行参数，便于切换不同实验设定。
    #####表示常用调整设置
"""

from pathlib import Path

# ────────── 刺激条件定义 ────────── #
CONDITION_NAMES = (
    "gain",      # 增益适应
    "dots",      # 随机点运动
    "flash",     # 明暗闪烁
    "taxis",     # 光趋性（holdout）
    "turning",   # 转向
    "position",  # 位置本体
    "open loop", # 被动无反馈
    "rotation",  # 旋转刺激
    "dark"       # 自发活动
)
# 每个条件的起止帧（左闭右开）
CONDITION_OFFSETS = (0, 649, 2422, 3078, 3735, 5047, 5638, 6623, 7279, 7879)
assert len(CONDITION_OFFSETS) == len(CONDITION_NAMES) + 1
# 训练条件编号
CONDITIONS_TRAIN   = (0, 1, 2, 4, 5, 6, 7, 8)
CONDITIONS_HOLDOUT = (3,)



# ────────── 数据维度 ────────── #
NUM_NEURONS = 71_721              # 神经元总数，trace.zarr主轴长度
COV_DIM     = 26                  # 刺激特征数（stimuli_features.zarr，每步26维特征，详见论文B.6）
STATIC_DIM  = 192                 # 空间静态特征数（position_embedding.zarr，每个神经元192维embedding）



# ────────── 时间窗参数 ────────── #
CONDITION_PADDING = 1             # 数据滑窗采样时的首尾padding步数
CONTEXT_LEN = 4                   ##### 输入上下文长度C，短上下文4，长上下文256
PRED_LEN    = 32                  # 预测步长
HORIZONS    = [1, 4, 8, 16, 32]   # 评估时各个步长
assert PRED_LEN in HORIZONS



# ────────── 划分比例 ────────── #
VAL_FRACTION  = 0.10               # 验证集占比
TEST_FRACTION = 0.20               # 测试集占比，剩余部分为训练集，holdout条件单独留作最终测试



# ────────── 训练 & 运行 ────────── #
SUBSET_SIZE   = 0                  ##### ≤0时用全部神经元，调试/开发用子集加速
CHUNK_SIZE    = 360                ##### 单次最大读入样本数，防OOM
BATCH_SIZE    = 10                 ##### 单个训练步处理的窗口数
LEARNING_RATE = 1e-3               # 主学习率
NUM_EPOCHS    = 50                 # 最大训练轮数，早停提前终止。一轮表示一次全窗口的训练。强制早停直接设置极大即可。
EARLY_STOPPING_DELTA = 1e-5        # 早停判断差值
EARLY_STOPPING_PATIENCE = 20       # 早停判断轮数
WEIGHT_DECAY  = 1e-4               # L2权重衰减
SEED          = 0                  ##### 全局随机种子
VAL_WINDOW_COUNT = 500             ##### 每轮验证完成训练的窗口数
VAL_SAMPLE_FRACTION = 0.2          ##### 只评估X%的验证窗口
VIDEO_CROP_XYZ = (64, 36, 9)       ##### 视频/体积模型取空间的子集，不高于当前分辨率



# ────────── 数据路径 ────────── #

# 本地路径
#DATA_ROOT = Path("/root/code/zapbench_data")
#TRACE_PATH   = DATA_ROOT / "traces.zarr"              # 神经元活动trace数据
#STIMULI_PATH = DATA_ROOT / "stimuli_features.zarr"    # 刺激特征
#STATIC_PATH  = DATA_ROOT / "position_embedding.zarr"  # 空间静态特征

# 云端数据路径
GCS_ROOT = "gs://zapbench-release/volumes/20240930"
TRACE_PATH   = f"{GCS_ROOT}/traces/"
STIMULI_PATH =  f"{GCS_ROOT}/stimuli_features/"
STATIC_PATH  =  f"{GCS_ROOT}/position_embedding/"
VIDEO_RES_TAG = "s2"                           # 分辨率, s0=(2048, 1152, 72)，s1=(1024, 576, 72), s2=(512, 288, 72)
VIDEO_PATH   = f"{GCS_ROOT}/df_over_f_xyz_chunked/{VIDEO_RES_TAG}"  
VIDEO_SEGMENTATION_PATH = f"{GCS_ROOT}/segmentation"
BRAIN_MASK_PATH         = f"{GCS_ROOT}/mask/" 

# 【数据获取】下载官方数据用如下命令：
#  gsutil -m cp -r gs://zapbench-release/volumes/20240930/*/



# ────────── 模型选择 ────────── #
# 可选: "linear", "tide", "tsmixer", "time_mix", "unet"
# - linear: 线性回归基线
# - tide: 时序深度预测（TiDE）
# - tsmixer: 时序混合器
# - time_mix: 时序混合/可扩展结构
# - unet: 空间-时序U-Net/体积预测
MODEL_NAME        = "unet"  



# ────────── 断点恢复与模型存档设置 ────────── #
RESUME_TRAIN = False           # True: 自动恢复训练；False: 每次从头开始
SAVE_CHECKPOINTS = True        # 是否启用训练模型存档
ONLY_TEST = False             # 是否直接进行测试
TEST_FROM_FINAL_OUTPUT = False  # True用最终存档，False用断点

VIDEO_CROP_TAG = f"crop_{VIDEO_CROP_XYZ[0]}x{VIDEO_CROP_XYZ[1]}x{VIDEO_CROP_XYZ[2]}"
subset_tag = f"subset{SUBSET_SIZE}" if SUBSET_SIZE > 0 else "full"
if MODEL_NAME == "unet":
    CHECKPOINT_ROOT = (
        Path("checkpoint") /
        MODEL_NAME /
        f"C{CONTEXT_LEN}" /
        VIDEO_RES_TAG /
        VIDEO_CROP_TAG /
        f"SEED{SEED}"
    )
    OUTPUT_ROOT = (
        Path("output") /
        MODEL_NAME /
        f"C{CONTEXT_LEN}" /
        VIDEO_RES_TAG /
        VIDEO_CROP_TAG /
        f"SEED{SEED}"
    )
else:
    CHECKPOINT_ROOT = (
        Path("checkpoint") /
        MODEL_NAME /
        f"C{CONTEXT_LEN}" /
        subset_tag /
        f"SEED{SEED}"
    )
    OUTPUT_ROOT = (
        Path("output") /
        MODEL_NAME /
        f"C{CONTEXT_LEN}" /
        subset_tag /
        f"SEED{SEED}"
    )
# 模型存档主目录，结构为 checkpoint/<MODEL_NAME>/C/...
# 实验最终输出主目录，结构为 output/<MODEL_NAME>/C/...
