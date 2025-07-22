#model/tsmix_model.py
"""
TSMixer 与 TimeMix 模型
------------------------------------------------------------------
    - 两个独立工厂函数:
        * build_tsmixer_model  : TimeMix + FeatureMix
        * build_time_mix_model : 仅 TimeMix 
"""

from typing import Any, Tuple, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct

from models.util import ReversibleInstanceNorm, activation_fn_from_str

# ================== 基础子模块 ==================

class Identity(nn.Module):
    """恒等层."""
    @nn.compact
    def __call__(self, x):
        return x

# ---- TimeMix ---- #
class TimeMixBlock(nn.Module):
    """时间混合块.
    输入形状: (B, T, F)
    流程: flatten(B,T*F) → (可选 norm, 此处 Identity) → reshape → 转置(B,F,T) → Dense(T→T) → 激活 → 转回(B,T,F) → Dropout → Residual
    参数:
        activation_fn : 激活函数.
        dropout       : dropout 概率.
        residual      : 是否使用残差.
    返回:
        jnp.ndarray: 形状 (B, T, F)
    """
    activation_fn: Any
    dropout: float
    residual: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        inputs = x
        B, T, F = x.shape
        x = Identity()(x.reshape(B, T * F))
        x = x.reshape(B, T, F)
        # 时间维混合: 转置到 (B,F,T)
        x = x.transpose(0, 2, 1)
        # Dense(T→T)
        x = nn.Dense(features=T)(x)
        x = self.activation_fn(x)
        # 回到 (B,T,F)
        x = x.transpose(0, 2, 1)
        # Dropout
        x = nn.Dropout(rate=self.dropout, deterministic=not train)(x)
        return x + inputs if self.residual else x

# ---- FeatureMix  ---- #
class FeatureMixBlock(nn.Module):
    """特征混合块.
    输入: (B, T, F)
    流程: flatten→(Identity norm)→reshape → [Dense(F→mlp_dim) + 激活 + Dropout] → [Dense(mlp_dim→F) + Dropout] → Residual
    参数:
        activation_fn : 激活函数.
        dropout       : dropout 概率.
        mlp_dim       : 瓶颈维度 (>0 时启用两段结构; 若 <=0 则退化为 Identity 残差)。
        residual      : 是否使用残差.
    返回:
        jnp.ndarray: (B, T, F)
    """
    activation_fn: Any
    dropout: float
    mlp_dim: int
    residual: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        inputs = x
        B, T, F = x.shape
        x = Identity()(x.reshape(B, T * F))
        x = x.reshape(B, T, F)
        if self.mlp_dim > 0:
            x = nn.Dense(features=self.mlp_dim)(x)  # F→mlp_dim
            x = self.activation_fn(x)
            x = nn.Dropout(rate=self.dropout, deterministic=not train)(x)
            x = nn.Dense(features=F)(x)             # mlp_dim→F
            x = nn.Dropout(rate=self.dropout, deterministic=not train)(x)
            return x + inputs if self.residual else x
        else:
            return inputs  # 退化: 不变换

# ---- MixerBlock (TimeMix + FeatureMix) ---- #
class MixerBlock(nn.Module):
    """单个混合块: 时间混合后接特征混合.
    参数:
        activation_fn : 激活函数.
        dropout       : dropout 概率.
        mlp_dim       : 特征瓶颈维度.
        time_residual : 时间分支残差.
        feat_residual : 特征分支残差.
    返回:
        jnp.ndarray: (B, T, F)
    """
    activation_fn: Any
    dropout: float
    mlp_dim: int
    time_residual: bool = True
    feat_residual: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        x = TimeMixBlock(
            activation_fn=self.activation_fn,
            dropout=self.dropout,
            residual=self.time_residual,
        )(x, train=train)
        x = FeatureMixBlock(
            activation_fn=self.activation_fn,
            dropout=self.dropout,
            mlp_dim=self.mlp_dim,
            residual=self.feat_residual,
        )(x, train=train)
        return x

# ================== TSMixer (Time + Feature) ==================
@struct.dataclass
class TSMixerConfig:
    """TSMixer 配置 .
    参数:
        pred_len : 预测步数 H.
        n_block  : 块数 .
        dropout  : dropout 概率.
        mlp_dim  : 特征瓶颈维度 .
        activation : 激活函数名.
        instance_norm : 是否应用可逆实例归一化.
        revert_instance_norm : 末尾是否还原.
    """
    pred_len: int = 1
    n_block: int = 5
    dropout: float = 0.1
    mlp_dim: int = 100
    activation: str = "relu"
    instance_norm: bool = True
    revert_instance_norm: bool = True

class TSMixer(nn.Module):
    """
    TSMixer 模型 (含时间与特征混合).
    输入: (B, C, F)  输出: (B, H, F)
    参数:
        cfg (TSMixerConfig): 配置.
    返回:
        jnp.ndarray: (B, pred_len, F)
    """
    cfg: TSMixerConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool=False) -> jnp.ndarray:
        act_fn = activation_fn_from_str(self.cfg.activation)
        if self.cfg.instance_norm:
            rin = ReversibleInstanceNorm()
            x, stats = rin(x)
        for i in range(self.cfg.n_block):
            x = MixerBlock(
                activation_fn=act_fn,
                dropout=self.cfg.dropout,
                mlp_dim=self.cfg.mlp_dim,
                name=f"block{i+1}",
            )(x, train=train)
        # Temporal projection: (B,C,F) -> (B,F,C) -> Dense(C→H) -> (B,H,F)
        x = x.transpose(0, 2, 1)
        x = nn.Dense(self.cfg.pred_len)(x)
        x = x.transpose(0, 2, 1)
        if self.cfg.instance_norm and self.cfg.revert_instance_norm:
            x = rin.revert(x, stats)
        return x

# ================== TimeMix Only (时间混合基线) ==================
@struct.dataclass
class TimeMixOnlyConfig:
    """TimeMix-only 配置.
    参数:
        num_outputs   : 预测步数 H.
        n_block       : 块数.
        dropout       : dropout 概率.
        activation    : 激活函数名.
        instance_norm : 是否使用可逆实例归一化.
        revert_instance_norm  : 是否末尾还原.
    """
    num_outputs: int
    n_block: int = 5
    dropout: float = 0.1
    activation: str = "relu"
    instance_norm: bool = True
    revert_instance_norm: bool = True

class TimeMixOnly(nn.Module):
    """仅时间混合模型.
    输入: (B, C, F) 输出: (B, H, F)
    参数:
        cfg (TimeMixOnlyConfig): 配置.
    返回:
        jnp.ndarray: (B, H, F)
    """
    cfg: TimeMixOnlyConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool=False) -> jnp.ndarray:
        act_fn = activation_fn_from_str(self.cfg.activation)
        if self.cfg.instance_norm:
            rin = ReversibleInstanceNorm()
            x, stats = rin(x)
        for i in range(self.cfg.n_block):
            x = TimeMixBlock(
                activation_fn=act_fn,
                dropout=self.cfg.dropout,
                residual=True,
                name=f"time_block{i+1}",
            )(x, train=train)
        x = x.transpose(0, 2, 1)
        x = nn.Dense(self.cfg.num_outputs)(x)
        x = x.transpose(0, 2, 1)
        if self.cfg.instance_norm and self.cfg.revert_instance_norm:
            x = rin.revert(x, stats)
        return x

# ================== 工厂函数（已对齐官方配置） ==================

def build_tsmixer_model(
    context_len: int,
    pred_len: int,
    seed: int,
    effective_F: Optional[int] = None,
) -> Tuple[TSMixer, Any]:
    """
    构造 TSMixer 模型并初始化参数。
    自动映射规则（不改变外部调用方式）:
        - 若 context_len <= 4 视为:
              n_block=2, mlp_dim=256, dropout=0.0, instance_norm=False
        - 否则为 long_context:
              n_block=2, mlp_dim=128, dropout=0.0, instance_norm=True
        - 其余字段使用官方默认: activation='relu', revert_instance_norm=True
        - time_mix_mlp_dim 固定等于 C
    参数:
        context_len   : 输入时间步长 C.
        pred_len      : 预测步数 H.
        seed          : 随机种子.
        effective_F   : 神经元数.
    返回:
        (TSMixer, FrozenDict): 模型实例与其参数字典（仅 'params' 部分）。
    """
    F = effective_F

    if context_len <= 4:  # short_context
        _n_block = 2
        _mlp_dim = 256
        _dropout = 0.0
        _instance_norm = False
    else:                 # long_context
        _n_block = 2
        _mlp_dim = 128
        _dropout = 0.0
        _instance_norm = True

    cfg = TSMixerConfig(
        pred_len=pred_len,
        n_block=_n_block,
        dropout=_dropout,
        mlp_dim=_mlp_dim,
        activation="relu",
        instance_norm=_instance_norm,
        revert_instance_norm=True,
    )
    model = TSMixer(cfg)
    rng = jax.random.PRNGKey(seed)
    dummy = jnp.zeros((1, context_len, F), dtype=jnp.float32)
    variables = model.init(rng, dummy, train=False)
    return model, variables["params"]


def build_time_mix_model(
    *,
    context_len: int,
    pred_len: int,
    seed: int = 0,
    effective_F: Optional[int] = None,
) -> Tuple[TimeMixOnly, Any]:
    """
    构造 TimeMix-only 模型并初始化参数。
    自动映射规则:
        - 官方 time_mix_only: n_block 固定 5
        - dropout 在官方配置中为 0.0
        - instance_norm: short_context(T<=4) 为 False: long_context(>4) 为 True
        - 其它: activation='relu', revert_instance_norm=True
    参数:
        context_len (int): 输入时间步长 T.
        pred_len (int): 预测步数 H.
        seed (int): 随机种子.
        effective_F (Optional[int]): 神经元数.
    返回:
        (TimeMixOnly, FrozenDict): 模型实例与其参数字典（仅 'params' 部分）。
    """
    F = effective_F

    if context_len <= 4:
        _instance_norm = False
    else:
        _instance_norm = True

    cfg = TimeMixOnlyConfig(
        num_outputs=pred_len,
        n_block=5,
        dropout=0.0,
        activation="relu",
        instance_norm=_instance_norm,
        revert_instance_norm=True,
    )
    model = TimeMixOnly(cfg)
    rng = jax.random.PRNGKey(seed)
    dummy = jnp.zeros((1, context_len, F), dtype=jnp.float32)
    variables = model.init(rng, dummy, train=False)
    return model, variables["params"]

