# dataset_loader.py
"""
数据集加载器
---------------------
    - 支持trace/stimuli/static数据的高效流式加载
    - 条件切分、滑窗、随机采样与批次生成
"""

from __future__ import annotations

from typing import Optional, Iterable
import numpy as np
from tqdm import tqdm

from config import VAL_SAMPLE_FRACTION 
from utils import _condition_bounds, _adjust_bounds, _get_store 


# —— 静态协变量加载 —— #
_static_cache = None
def load_static(path: str, subset: Optional[np.ndarray]) -> np.ndarray:
    """
    加载静态协变量数据，缓存并返回。
    参数:
    - path: 静态协变量文件的路径。
    - subset: 神经元子集索引。
    返回:
    - 加载的静态协变量数据，形状为 (神经元数, 特征数)。
    """
    global _static_cache
    if _static_cache is None:
        tqdm.write("Loading static covariates...")
        store = _get_store(path)
        if subset is None:
            arr = store.read().result()
        else:
            # 直接在 TensorStore 层做切片，再 read
            arr = store[:, subset].read().result()
        _static_cache = arr
        tqdm.write(f"Static covariates loaded, shape={arr.shape}")
    return _static_cache




# —— 流式数据加载 —— #
def stream_loader(
    *,
    conditions: list[int],
    C: int,
    H: int,
    trace_path: str,
    stimuli_path: Optional[str],
    split: str,
    subset: Optional[np.ndarray] = None,
    chunk_size: Optional[int] = None,
    is_tide: bool = False,
    batch_size: Optional[int] = None,
    seed: int = 0,
) -> Iterable:
    """
    纯流式加载，在 train 时做全局随机混合。
    验证集可采样加速评估。
    参数:
    - conditions: 刺激条件的编号列表。
    - C: 上下文长度。
    - H: 预测步长。
    - trace_path: 神经元活动数据的路径。
    - stimuli_path: 刺激数据的路径。
    - split: 数据集切分方式。
    - subset: 神经元子集索引。
    - chunk_size: 每次加载数据块的大小。
    - is_tide: 是否启用 TiDE 模式。
    - batch_size: 每个批次的大小。
    - seed: 随机种子。
    返回:
    - Iterable: 返回一个数据批次的数据窗口。
    """

    # 1) 打开 stores & 预加载 static（TiDE 模式）
    trace_store = _get_store(trace_path)
    stim_store  = _get_store(stimuli_path) if (is_tide and stimuli_path) else None

    win_len = C + H
    overlap = win_len - 1

    # 2) 为每个条件计算 bounds、step，并生成 (cond, win_idx) 列表
    all_pairs: list[tuple[int, int]] = []
    cond_bounds: dict[int, tuple[int,int]] = {}
    cond_steps:  dict[int, int] = {}
    for cond in conditions:
        inc, exc = _adjust_bounds(*_condition_bounds(cond), C, split)
        total    = exc - inc
        step_cond = chunk_size or total
        cond_bounds[cond] = (inc, exc)
        cond_steps[cond]  = step_cond
        nwin = max(0, total - win_len + 1)
        all_pairs += [(cond, i) for i in range(nwin)]

    # 3.1) 验证集可采样加速
    if split == "val" and VAL_SAMPLE_FRACTION < 1.0:
        nsample = max(1, int(len(all_pairs) * VAL_SAMPLE_FRACTION))
        rng = np.random.default_rng(seed)
        all_pairs = rng.choice(all_pairs, size=nsample, replace=False)
        all_pairs = list(all_pairs)  # numpy数组转回list，保持后续兼容

    # 3.2) train 时全局打乱
    if split == "train":
        rng = np.random.default_rng(seed)
        rng.shuffle(all_pairs)

    # 4) 按 (cond, blk) 缓存 I/O 块
    block_cache: dict[tuple[int,int], tuple[int, np.ndarray, Optional[np.ndarray]]] = {}

    def load_one(cond: int, win_idx: int):
        """
        加载一个窗口的数据，根据条件和窗口索引读取数据。
        参数:
        - cond   : 当前处理的刺激条件编号。
        - win_idx: 当前条件下的窗口索引。
        返回:
        - (Xw, Yw) 或 (Xw, Xp, Xf, Yw):
        - Xw : 输入数据，形状为 (C,)。
        - Yw : 输出数据，形状为 (H,)。
        - Xp : 过去 C 步的刺激数据，形状为 (C,)。
        - Xf : 未来 H 步的刺激数据，形状为 (H,)。
        """
        inc, exc   = cond_bounds[cond]
        step_cond  = cond_steps[cond]
        start      = inc + win_idx
        blk        = win_idx // step_cond
        cache_key  = (cond, blk)

        if cache_key not in block_cache:
            b0 = inc + blk * step_cond
            b1 = min(b0 + step_cond + overlap, exc)
            if subset is not None:
                arr = trace_store[b0:b1, subset].read().result()
            else:
                arr = trace_store[b0:b1].read().result()
            sarr = (stim_store[b0:b1].read().result() if is_tide else None)
            block_cache[cache_key] = (b0, arr, sarr)

        b0, arr, sarr = block_cache[cache_key]
        rel = start - b0
        w   = arr[rel : rel + win_len]
        Xw  = w[:C]
        Yw  = w[C:]

        if not is_tide:
            return Xw, Yw

        s   = sarr[rel : rel + win_len] # 切片获取当前窗口的刺激数据
        Xp  = s[:C]
        Xf  = s[C:]
        return Xw, Xp, Xf, Yw  # 返回神经元活动数据和刺激数据

    # 5) 按 batch_size 聚合或逐窗口 yield
    if batch_size is None:
        for cond, idx in all_pairs:
            yield load_one(cond, idx)
    else:
        buf: list = []
        for cond, idx in all_pairs:
            buf.append(load_one(cond, idx))
            if len(buf) == batch_size:
                arrays = list(zip(*buf))
                yield tuple(np.stack(a, axis=0) for a in arrays)
                buf.clear()
        if buf:
            arrays = list(zip(*buf))
            yield tuple(np.stack(a, axis=0) for a in arrays)
