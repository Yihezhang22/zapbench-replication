#utils.py
"""
通用工具模块
---------------------
    1. 管理训练存档和最终输出存档
    2. 数据集窗口的条件边界、train/val/test分割索引
    3. TensorStore数据流式加载与本地/云端自动缓存
"""

import flax
from tqdm import tqdm
import tensorstore as ts
from pathlib import Path
from urllib.parse import urlparse

from config import CONDITION_OFFSETS, CONDITION_PADDING, VAL_FRACTION, TEST_FRACTION


# —— 存档路径与参数序列化 —— #
def get_dir(base_path):
    """
    根据传入的根路径生成存档目录，确保该目录存在。
    参数:
    - base_path: CHECKPOINT_ROOT 或 OUTPUT_ROOT。
    返回:
    - Path: 返回创建或已存在的目录路径。
    """
    base_path = Path(base_path)                   # 将基础路径转为 Path 对象
    base_path.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    return base_path


def save_params(path, params):
    """
    保存Flax模型参数到指定文件。
    参数:
    - path: 模型参数保存的文件路径。
    - params: Flax模型的参数。
    返回:
    - None: 将模型参数保存到指定路径。
    """
    path = path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(flax.serialization.to_bytes(params))
    tqdm.write(f"[Info] Saved model parameters to: {path}")

def save_train_state(path, state):
    """
    保存训练状态到指定文件(flax序列化格式, 含参数、优化器、步数等)
    参数:
    - path: 训练状态保存的文件路径。
    - state: Flax训练状态对象, 包含参数、优化器状态、当前步数等信息。  
    返回:
    - None: 将训练状态保存到指定路径。
    """
    path = path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(flax.serialization.to_bytes(state))
    tqdm.write(f"[Info] Saved training state to: {path}")


def load_params(path, params_template):
    """
    加载模型参数(如无效则返回模板)
    参数:
    - path: 模型参数的文件路径。
    - params_template: 默认的模型参数模板。
    返回:
    - 加载的模型参数，或者在加载失败时返回默认模板。
    """
    path = path
    if not path.exists() or path.stat().st_size < 16:
        tqdm.write(f"[Warning] No valid checkpoint at {path}, using fresh params.")
        return params_template
    try:
        with open(path, "rb") as f:
            data = f.read()
        tqdm.write(f"[Info] Successfully loaded params from: {path}")
        return flax.serialization.from_bytes(params_template, data)
    except Exception as e:
        tqdm.write(f"[Warning] Failed to load params from {path}: {e}")
        return params_template

def load_train_state(path, state_template):
    """
    加载训练状态(如无效则返回模板) 
    参数:
    - path: 训练状态的文件路径。
    - state_template: 默认的训练状态模板。
    返回:
    - 加载的训练状态，或者在加载失败时返回默认模板。
    """
    path = path
    if not path.exists() or path.stat().st_size < 16:
        tqdm.write(f"[Warning] No valid train state at {path}, using fresh state.")
        return state_template
    try:
        with open(path, "rb") as f:
            tqdm.write(f"[Info] Successfully loaded train state from: {path}")
            return flax.serialization.from_bytes(state_template, f.read())
    except Exception as e:
        tqdm.write(f"[Warning] Failed to load train state from {path}: {e}")
        return state_template



# —— 数据窗口边界与分割 —— #
def _condition_bounds(c: int) -> tuple[int, int]:
    """
    返回第c个条件的有效采样窗口区间 (已做padding)     
    参数:
    - c: 刺激条件的编号。
    返回:
    - tuple: 该条件的有效采样窗口区间(起始帧和结束帧), 包括了padding。
    """
    s = CONDITION_OFFSETS[c] + CONDITION_PADDING
    e = CONDITION_OFFSETS[c+1] - CONDITION_PADDING
    return s, e


def _adjust_bounds(inc: int, exc: int, C: int, split: str) -> tuple[int, int]:
    """
    按split类型将区间[inc, exc)切分为子集窗口区间, 验证集和测试集向前取得上下文
    参数:
    - inc  : 区间的起始位置。
    - exc  : 区间的结束位置。
    - C    : 上下文长度。
    - split: 数据切分类型。
    返回:
    - tuple[int, int]: 划分后的子集窗口区间。
    """
    assert exc - inc >= C + 1, "segment too short"
    total     = exc - inc
    test_len  = int(total * TEST_FRACTION)
    val_len   = int(total * VAL_FRACTION)
    train_len = total - test_len - val_len

    if split == "train":
        return inc, inc + train_len 
    if split == "val":
        return inc + train_len - C, inc + train_len + val_len
    if split == "test":
        return inc + train_len + val_len - C, exc
    if split == "test_holdout":
        return inc, exc

    raise ValueError(f"unknown split: {split}")



# —— TensorStore 数据流加载与本地/云端缓存 —— #
_ts_store_cache: dict[str, ts.TensorStore] = {}

def _get_store(path: str) -> ts.TensorStore:
    """
    以本地或 GCS 路径加载 TensorStore 对象，兼容 zarr3 & zarr2 协议，并自动缓存。
    支持本地文件系统和 gs://bucket/path 两种格式。
    """
    key = str(path)
    if key not in _ts_store_cache:
        # GCS 后端
        if key.startswith("gs://"):
            # 解析 "gs://bucket/inner/path"
            parsed = urlparse(key)
            bucket = parsed.netloc
            inner_path = parsed.path.lstrip('/')
            # 优先尝试 zarr3，然后回退到 zarr2
            spec3 = {
                'driver': 'zarr3',
                'kvstore': {
                    'driver': 'gcs',
                    'bucket': bucket,
                    'path': inner_path
                },
                'open': True
            }
            spec2 = {
                'driver': 'zarr',
                'kvstore': {
                    'driver': 'gcs',
                    'bucket': bucket,
                    'path': inner_path
                },
                'open': True
            }
            try:
                store = ts.open(spec3).result()
            except Exception:
                store = ts.open(spec2).result()
        else:
            # 本地文件系统
            spec3 = {
                'driver': 'zarr3',
                'kvstore': {'driver': 'file', 'path': key},
                'open': True
            }
            spec2 = {
                'driver': 'zarr',
                'kvstore': {'driver': 'file', 'path': key},
                'open': True
            }
            try:
                store = ts.open(spec3).result()
            except Exception:
                store = ts.open(spec2).result()
        _ts_store_cache[key] = store
    return _ts_store_cache[key]
