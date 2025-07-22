# test_utils.py
"""
模型评测脚本（时序）
---------------------
    - 对训练完成后的模型在所有测试/holdout条件和各horizon下进行MAE评测
    - 支持指定加载断点存档或最终output存档
"""

import numpy as np
import jax
from tqdm import tqdm
from typing import Optional

from models import get_model_by_name
from dataset_loader import stream_loader, load_static
from utils import load_params, get_dir
from config import (
    # —— 基础实验设置 ——
    MODEL_NAME, TEST_FROM_FINAL_OUTPUT,
    # —— 刺激条件定义 ——
    CONDITION_NAMES, CONDITIONS_HOLDOUT,
    # —— 数据维度 ——
    NUM_NEURONS, 
    # —— 时间窗参数 ——
    CONTEXT_LEN, PRED_LEN, HORIZONS,
    # —— 训练与运行参数 ——
    BATCH_SIZE, CHUNK_SIZE, SEED,
    # —— 数据路径 ——
    TRACE_PATH, STIMULI_PATH, STATIC_PATH,
    # —— 存档路径 ——
    CHECKPOINT_ROOT, OUTPUT_ROOT,
)


def test_and_report(subset: Optional[np.ndarray]) -> dict:
    """
    全条件/全步长评测主函数
    -------------------
    - 自动加载最佳存档/最终存档参数
    - 输出各条件各horizon的MAE
    参数：
    - subset:  神经元子集的索引
    返回：
    - dict  : 包含各条件和各步长的MAE评测结果。
    """
    if subset is None:
        tqdm.write("[INFO] Evaluating on: whole brain")
    else:
        tqdm.write(f"[INFO] Evaluating on subset: {len(subset)}")

    # 判断是否使用 TiDE 模型
    is_tide = MODEL_NAME.lower() == "tide"

    C = CONTEXT_LEN
    Hmax = PRED_LEN
    batch = BATCH_SIZE

    # ------ 选择模型参数路径 ------
    if TEST_FROM_FINAL_OUTPUT:
        params_path = OUTPUT_ROOT / "best_params.npz"
    else:
        params_path = CHECKPOINT_ROOT / "best_params.npz"

    # ------ 加载模型 ------
    if is_tide:
        Xs = load_static(STATIC_PATH, subset)
        F_cur = NUM_NEURONS if subset is None else len(subset)
        model, _ = get_model_by_name(
            name=MODEL_NAME,
            context_len=C,
            pred_len=Hmax,
            seed=SEED,
            effective_F=F_cur,
        )
    else:
        F_cur = NUM_NEURONS if subset is None else len(subset)
        model, _ = get_model_by_name(
            name=MODEL_NAME,
            context_len=C,
            pred_len=Hmax,
            seed=SEED,
            effective_F=F_cur,
        )

    # 加载参数
    params_template = None
    params = load_params(params_path, params_template)
    if params is None:
        raise ValueError(f"Failed to load parameters from {params_path}")


    # ------ 测试主循环 ------
    results = {}
    for cid, name in enumerate(CONDITION_NAMES):
        split = "test_holdout" if cid in CONDITIONS_HOLDOUT else "test"

        if is_tide:
            # 如果是TiDE模型，加载静态协变量数据
            Xs = load_static(STATIC_PATH, subset)

        total_mae_per_h = None
        total_count = 0

        # 加载数据
        loader = stream_loader(
            conditions=[cid],
            C=C, H=Hmax,
            trace_path=TRACE_PATH,
            stimuli_path=STIMULI_PATH,
            split=split,
            subset=subset,
            chunk_size=CHUNK_SIZE,
            is_tide=is_tide,
            batch_size=batch,
            seed=SEED,
        )
        for batch_data in loader:
            if is_tide:
                # TiDE模型的前向计算
                xb, xp, xf, yb = batch_data
                xs = Xs
                pred = model.apply(
                    {"params": params},
                    xb, xs, xp, xf,
                    train=False,
                )
            else:
                # 常规模型的前向计算
                xb, yb = batch_data
                pred = model.apply(
                    {"params": params},
                    xb,
                    train=False,
                )

            # 计算当前批次的MAE
            batch_mae_each = np.mean(np.abs(pred - yb), axis=(0, 2))
            b = yb.shape[0]
            if total_mae_per_h is None:
                total_mae_per_h = batch_mae_each * b
            else:
                total_mae_per_h += batch_mae_each * b
            total_count += b

        # 计算每个条件和步长的平均MAE
        mae_each = total_mae_per_h / total_count
        results[name] = {}
        for h in HORIZONS:
            mae = float(mae_each[h - 1])  # 对每个horizon计算MAE
            tqdm.write(f"{name:12s} H={h:2} → MAE {mae:.5f}")
            results[name][f"h={h}"] = mae

    # 返回评测结果
    return results
