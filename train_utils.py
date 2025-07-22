# train_utils.py
"""
训练与评估主流程工具
---------------------
    - 统一实现所有时序模型(Linear/TiDE/TSMixer/Time_mix)的训练、验证与测试主循环
    - 包含训练步、流式验证评估、早停与存档
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
import random
from tqdm import tqdm
from typing import Callable, Optional

from models import get_model_by_name
from dataset_loader import stream_loader, load_static
from utils import (
    get_dir, 
    save_params, save_train_state,
    load_params, load_train_state,
    _adjust_bounds, _condition_bounds,
)
from models.util import TrainState
from config import (
    # —— 基础实验设置 ——
    MODEL_NAME, RESUME_TRAIN, SAVE_CHECKPOINTS,
    # —— 刺激条件定义 ——
    CONDITIONS_TRAIN, 
    # —— 数据维度 ——
    NUM_NEURONS,
    # —— 时间窗参数 ——
    CONTEXT_LEN, PRED_LEN,
    # —— 训练与运行参数 ——
    CHUNK_SIZE, BATCH_SIZE, NUM_EPOCHS,
    EARLY_STOPPING_DELTA, EARLY_STOPPING_PATIENCE,
    WEIGHT_DECAY, SEED, VAL_WINDOW_COUNT, LEARNING_RATE,
    # —— 数据路径 ——
    TRACE_PATH, STIMULI_PATH, STATIC_PATH,
    # —— 存档路径 ——
    CHECKPOINT_ROOT, OUTPUT_ROOT,
)


def make_train_step(is_tide: bool) -> Callable:
    """
    构建训练单步函数(JIT加速),兼容TiDE与常规时序模型。
    - 自动按is_tide选择输入和模型前向方式
    - 按MAE定义损失
    - 保证dropout_key按step递进
    返回: JIT后的(step_fn),输入当前TrainState和batch数据,返回新state和loss
    """
    def step(state, xb, *args):
        # 为当前step生成独立Dropout随机数
        dropout_rng = jax.random.fold_in(state.dropout_key, state.step)

        def loss_fn(params):
            if is_tide:
                xb_, xs, xp, xf, yb = xb, *args
                # TiDE模型前向与损失
                pred = state.apply_fn(
                    {'params': params},
                    xb_, xs, xp, xf,
                    train=True,
                    rngs={'dropout': dropout_rng},
                )
                loss = jnp.mean(jnp.abs(pred - yb))
            else:
                xb_, yb = xb, args[0]
                # 常规模型前向与损失
                pred = state.apply_fn(
                    {'params': params},
                    xb_,
                    train=True,
                    rngs={'dropout': dropout_rng},
                )
                loss = jnp.mean(jnp.abs(pred - yb))
            return loss

        # 自动反向传播与状态更新
        loss, grad = jax.value_and_grad(loss_fn)(state.params)
        new_state = state.apply_gradients(grads=grad)
        # 更新dropout随机种子
        new_state = new_state.replace(dropout_key=jax.random.split(dropout_rng)[0])
        return new_state, loss

    return jax.jit(step)


# ========== 验证集流式评估函数 ==========
def val_stream_eval(
    state,
    model,
    is_tide,
    C,
    Hmax,
    trace_path,
    stimuli_path: Optional[str] = None,
    static_path: Optional[str] = None,
    subset: Optional[np.ndarray] = None
):
    """
    在验证集上流式评估模型平均MAE.支持TiDE和常规模型。
    - 按每个训练条件遍历所有验证窗口
    - 动态累加所有步长(Hmax)下的绝对误差
    参数：
    - state: 当前训练的状态，包含模型参数、优化器状态等。
    - model: 待评估的模型。
    - is_tide: 是否为 TiDE 模型。
    - C: 上下文长度。
    - Hmax: 最大预测步长。
    - trace_path, stimuli_path, static_path: 数据文件路径
    - subset: 神经元子集的索引
    返回：
    - val_mae: 整体的验证集平均 MAE。
    """
    total_mae_per_h = None  # 所有horizon步长下绝对误差和
    total_count = 0         # 总窗口数

    for cid in CONDITIONS_TRAIN:
        # TiDE模型提前加载静态协变量
        if is_tide:
            Xs = load_static(static_path, subset)
        loader = stream_loader(
            conditions=[cid],
            C=C, H=Hmax,
            trace_path=trace_path,
            stimuli_path=stimuli_path,
            split="val",
            subset=subset,
            chunk_size=CHUNK_SIZE,
            is_tide=is_tide,
            batch_size=BATCH_SIZE,
            seed=SEED,
        )
        for batch_data in loader:
            if is_tide:
                xb, xp, xf, yb = batch_data
                xs = Xs
                pred = model.apply(
                    {"params": state.params},
                    xb, xs, xp, xf,
                    train=False,
                )
            else:
                xb, yb = batch_data
                pred = model.apply(
                    {"params": state.params},
                    xb,
                    train=False,
                )

            # 对每步长（horizon）分别计算MAE并累计
            batch_mae_each = np.mean(np.abs(pred - yb), axis=(0, 2))
            b = yb.shape[0]
            if total_mae_per_h is None:
                total_mae_per_h = batch_mae_each * b
            else:
                total_mae_per_h += batch_mae_each * b
            total_count += b

    mae_each = total_mae_per_h / total_count   # 各步长平均MAE
    val_mae = float(np.mean(mae_each))         # 所有步长总体MAE
    return val_mae


# ========== 主函数 ==========
def train_and_validate(subset: Optional[np.ndarray]):
    """
    训练与验证主循环
    -------------------
    - 支持 TiDE 与常规模型
    - 按指定步长训练模型，并在每轮训练后进行验证集流式评估
    - 支持早停机制：基于验证集 MAE 判断是否提前停止训练，避免过拟合
    参数：
    - subset: 神经元子集的索引。
    """
    if subset is None:
        tqdm.write("[INFO] Training on: whole brain")
    else:
        tqdm.write(f"[INFO] Training on subset: {len(subset)}")
    
    is_tide = MODEL_NAME.lower() == "tide"

    random.seed(SEED)
    np.random.seed(SEED)
    C = CONTEXT_LEN
    Hmax = PRED_LEN
    batch = BATCH_SIZE
    tqdm.write(f"[Batch] fixed batch size = {batch}")

    # 计算每 epoch 步数 
    total_windows = 0
    for cid in CONDITIONS_TRAIN:
        inc, exc = _adjust_bounds(*_condition_bounds(cid), C, split="train")
        total_windows += max(0, (exc - inc) - (C + Hmax) + 1)
    steps_per_epoch = max(1, total_windows // batch)
    total_steps = steps_per_epoch * NUM_EPOCHS

    # 如果是TiDE模型，加载静态数据
    if is_tide:
        Xs  = load_static(STATIC_PATH, subset)

    # ----------------- 模型与状态 -----------------
    if is_tide:
        model, params = get_model_by_name(
            name=MODEL_NAME,
            context_len=C,
            pred_len=Hmax,
            seed=SEED,
            effective_F=Xs.shape[0],
        )
    else:
        F_cur = NUM_NEURONS if subset is None else len(subset)
        model, params = get_model_by_name(
            name=MODEL_NAME,
            context_len=C,
            pred_len=Hmax,
            seed=SEED,
            effective_F=F_cur,
        )
    # 创建训练状态，包含模型参数、优化器和dropout种子
    state_template = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adamw(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
        dropout_key=jax.random.split(jax.random.PRNGKey(SEED))[0], 
    )
    # 如果需要恢复训练，从存档中加载模型和训练状态
    if RESUME_TRAIN and SAVE_CHECKPOINTS:
        state = load_train_state(CHECKPOINT_ROOT / "latest_state.msgpack", state_template)
        best_params = load_params(CHECKPOINT_ROOT / "best_params.npz", state.params)
        best_mae = val_stream_eval(state.replace(params=best_params), model, is_tide, C, Hmax, TRACE_PATH, STIMULI_PATH, STATIC_PATH, subset)
        tqdm.write(f"[Resume] Loaded best MAE from checkpoint: {best_mae:.5f}")
        no_imp = 0
    else:
        state = state_template
        best_params = state.params
        best_mae, no_imp = float("inf"), 0
    # 将训练状态移动到适当的设备
    state = jax.device_put(state)

    # ----------------- 训练主循环 -----------------
    train_step = make_train_step(is_tide)
    step = 0
    val_every_steps = max(1, VAL_WINDOW_COUNT // batch)

    pbar = tqdm(total=total_steps, desc="Training Steps", ncols=100)
    early_stopped = False  # 标记是否早停

    for epoch in range(NUM_EPOCHS):
        loader = stream_loader(
            conditions=CONDITIONS_TRAIN,
            C=C, H=Hmax,
            trace_path=TRACE_PATH,
            stimuli_path=STIMULI_PATH,
            split="train",
            subset=subset,
            chunk_size=CHUNK_SIZE,
            is_tide=is_tide,
            batch_size=batch,
            seed=SEED,
        )
        for batch_data in loader:
            if is_tide:
                xb, xp, xf, yb = batch_data
                xs = Xs
                state, loss = train_step(state, xb, xs, xp, xf, yb)
            else:
                xb, yb = batch_data
                state, loss = train_step(state, xb, yb)

            step += 1
            pbar.set_postfix({"loss": float(loss)})
            pbar.update(1)

            if step % val_every_steps == 0:
                val_mae = val_stream_eval(state, model, is_tide, C, Hmax, TRACE_PATH, STIMULI_PATH, STATIC_PATH, subset)
                tqdm.write(f"[Val {step}] MAE={val_mae:.5f}")
                pbar.set_postfix({"loss": float(loss), "val_mae": val_mae})
                if val_mae < best_mae - EARLY_STOPPING_DELTA:
                    best_mae, no_imp, best_params = val_mae, 0, state.params
                    save_params(get_dir(CHECKPOINT_ROOT) / "best_params.npz", best_params)
                    save_train_state(get_dir(CHECKPOINT_ROOT) / "latest_state.msgpack", state)
                    tqdm.write(f"[Info] Best parameters at step {step}.")
                else:
                    no_imp += 1
                    if no_imp >= EARLY_STOPPING_PATIENCE:
                        tqdm.write(f"[Info] Early stopping at step {step}.")
                        state = state.replace(params=best_params)
                        early_stopped = True
                        break
        if early_stopped:
            break
    
    if early_stopped:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        save_params(get_dir(OUTPUT_ROOT) / "best_params.npz", best_params)
        save_train_state(get_dir(OUTPUT_ROOT) / "latest_state.msgpack", state)
        tqdm.write(f"[Info] Saved final output to:: {OUTPUT_ROOT}")

    tqdm.write("训练完成！")
