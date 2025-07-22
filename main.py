# main.py
"""
主脚本
=========================
* 根据 config.MODEL_NAME 调用
  - 时序模型  → train_utils / test_utils
  - UNet     → train_unet    / test_unet
* 支持 ONLY_TEST / TEST_FROM_FINAL_OUTPUT
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]          = "0"          # 单卡
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]   = "platform"

import json
import numpy as np
import jax

from config import (
    MODEL_NAME, ONLY_TEST, TEST_FROM_FINAL_OUTPUT,
    SUBSET_SIZE, NUM_NEURONS, SEED,
    CONTEXT_LEN,
    VIDEO_RES_TAG, VIDEO_CROP_TAG, subset_tag
)

# 动态导入
if MODEL_NAME.lower() == "unet":
    from train_unet    import train_and_validate_unet as _trainer
    from test_unet     import test_and_report_unet   as _tester
else:
    from train_utils import train_and_validate   as _trainer
    from test_utils  import test_and_report      as _tester


def build_subset_for_timeseries():
    """仅时序模型可选神经元子集, UNet 返回 None。"""
    if MODEL_NAME.lower() == "unet":
        return None
    F_total = NUM_NEURONS
    if 0 < SUBSET_SIZE < F_total:
        rng = np.random.default_rng(SEED)
        return rng.choice(F_total, SUBSET_SIZE, replace=False)
    if SUBSET_SIZE > F_total:
        print(f"[WARN] SUBSET_SIZE > {F_total}, 使用全量")
    return None


def main() -> None:
    print("JAX devices:", jax.devices())
    subset = build_subset_for_timeseries()

    # 决定结果文件前缀
    prefix = "output" if TEST_FROM_FINAL_OUTPUT else "checkpoint"
    if MODEL_NAME.lower() == "unet":
        result_fname = (
            f"{MODEL_NAME}_C{CONTEXT_LEN}_{VIDEO_RES_TAG}_{VIDEO_CROP_TAG}_"
            f"seed{SEED}_{prefix}.json"
        )
    else:
        result_fname = (
            f"{MODEL_NAME}_C{CONTEXT_LEN}_{subset_tag}_"
            f"seed{SEED}_{prefix}.json"
        )

    # ——— 流程分支 ——— #
    if MODEL_NAME.lower() == "unet":
        # UNet: trainer/tester 均不接参数
        if ONLY_TEST:
            test_results = _tester()
        else:
            _trainer()
            test_results = _tester()
    else:
        # 时序模型：trainer/tester 接 subset（可能为 None）
        if ONLY_TEST:
            test_results = _tester(subset)
        else:
            _trainer(subset)
            test_results = _tester(subset)

    # 保存 JSON 结果
    with open(result_fname, "w") as fp:
        json.dump(test_results, fp, indent=2)
    print(f"✅ Done. Results saved → {result_fname}")


if __name__ == "__main__":
    main()
