# models/util.py
"""
模型通用工具函数与结构
---------------------
    - 提供实例归一化、激活函数选择等辅助类与函数
    - 扩展Flax训练状态(如管理dropout随机种子)
    - 供各模型及主流程模块直接复用
"""

import flax
from flax.training import train_state
import jax.numpy as jnp
from typing import Tuple, Any, Sequence
import flax.linen as nn


# —— 可逆实例归一化 —— #
class ReversibleInstanceNorm:
    """
    可逆实例归一化，对 (B, T, F) 张量做规范化并可恢复。
    """
    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        # x: (B, T, F)
        mean = x.mean(axis=1, keepdims=True)   # (B,1,F)
        std  = x.std(axis=1,  keepdims=True)   # (B,1,F)
        y = (x - mean) / (std + self.eps)
        return y, (mean, std)

    def revert(self, x: jnp.ndarray, stats: Tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        mean, std = stats
        return x * (std + self.eps) + mean


# —— 激活函数选择工具 —— #
def activation_fn_from_str(name: str):
    """
    根据名称选择激活函数。
    """
    name = name.lower()
    if name == "relu":
        from flax.linen import relu
        return relu
    if name == "gelu":
        from flax.linen import gelu
        return gelu
    if name == "swish":
        from flax.linen import swish
        return swish
    raise ValueError(f"Unknown activation: {name}")

# —— Flax训练状态子类 —— #
@flax.struct.dataclass
class TrainState(train_state.TrainState):
    """
    Flax训练状态扩展
    """
    dropout_key: Any  # 用于管理dropout等操作的JAX PRNGKey


# —— 基础MLP残差块 —— #
class MLPResidual(nn.Module):
    """
    多层感知机(MLP)残差块, 支持可选dropout、LayerNorm和残差连接。
    输入输出shape与inputs一致。
    """
    activation_fn: Any
    num_hidden: int
    num_output: int
    dropout_prob: float = 0.0
    layer_norm: bool = False
    use_residual: bool = True

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, train: bool) -> jnp.ndarray:
        x = nn.Dense(features=self.num_hidden)(inputs)
        x = self.activation_fn(x)
        x = nn.Dense(features=self.num_output)(x)
        x = nn.Dropout(rate=self.dropout_prob, deterministic=not train)(x)
        if self.use_residual:
            x = x + nn.Dense(features=self.num_output)(inputs)
        if self.layer_norm:
            x = nn.LayerNorm()(x)
        return x


# —— 多层堆叠MLP残差块 —— #
class StackedMLPResidual(nn.Module):
    """
    按num_hiddens配置堆叠多个MLP残差块, 支持可选dropout、LayerNorm与残差连接。
    用于深层特征投影/解码。
    """
    activation_fn: Any
    num_hiddens: Sequence[int]
    dropout_prob: float = 0.0
    layer_norm: bool = False
    use_residual: bool = True

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, train: bool) -> jnp.ndarray:
        x = inputs
        for i in range(len(self.num_hiddens) - 1):
            x = MLPResidual(
                activation_fn=self.activation_fn,
                num_hidden=self.num_hiddens[i],
                num_output=self.num_hiddens[i + 1],
                dropout_prob=self.dropout_prob,
                layer_norm=self.layer_norm,
                use_residual=self.use_residual,
            )(x, train)
        return x
