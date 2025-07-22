# models/tide_model.py
"""
TiDE模型结构与工厂函数
---------------------
    - 定义TiDE时序预测模型
    - 提供build_tide_model工厂函数统一模型构建与参数初始化
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Tuple, Sequence
from flax import struct

from models.util import(
    ReversibleInstanceNorm, activation_fn_from_str, 
    MLPResidual, StackedMLPResidual
)
from config import COV_DIM, STATIC_DIM, PRED_LEN 


# —— 配置类 —— #
@struct.dataclass
class TideConfig:
    """
    TiDE模型配置类, 集中管理所有结构、归一化、消融与正则相关参数。
    """
    pred_len: int = PRED_LEN                 #预测步长 (H)
    past_covariates_num_hidden: int = 128    #过去协变量的隐藏层神经元数目
    past_covariates_dim: int = 32            #过去协变量的输入维度
    future_covariates_num_hidden: int = 128  #未来协变量的隐藏层神经元数目
    future_covariates_dim: int = 32          #未来协变量的输入维度
    encoder_decoder_num_hiddens: Sequence[int] = (128, 128)  #编码器和解码器的隐藏层神经元数目。
    decoder_dim: int = 32                    #解码器的输出维度
    temporal_decoder_num_hidden: int = 128   #时序解码器的隐藏层神经元数目
    activation: str = "relu"                 #激活函数类型
    dropout_prob: float = 0.0                #Dropout 概率, 用于防止过拟合, 在训练阶段启用。
    layer_norm: bool = False                 #是否启用 LayerNorm 层。在某些模型中，启用可提高训练稳定性。False
    use_residual: bool = True                #是否使用残差连接。残差连接有助于深度网络的训练，避免梯度消失。True
    instance_norm: bool = False              #是否启用可逆实例归一化。实例归一化有助于减小不同数据之间的尺度差异。False
    revert_instance_norm: bool = False       #是否恢复实例归一化。在某些模型中，反归一化有助于在训练后恢复原始数据分布。False 
    ablate_past_timeseries: bool = False     #是否消融过去时序数据 False
    ablate_static_covariates: bool = True    #是否消融静态协变量 True
    ablate_past_covariates: bool = True      #是否消融过去协变量 True
    ablate_future_covariates: bool = False   #是否消融未来协变量 False



# —— 主模型类 —— #
class Tide(nn.Module):
    config: TideConfig

    def setup(self):
        cfg = self.config
        act = activation_fn_from_str(cfg.activation)
        # 子模块实例化
        # 过去协变量投影层
        self.past_proj = MLPResidual(
            activation_fn=act,
            num_hidden=cfg.past_covariates_num_hidden,
            num_output=cfg.past_covariates_dim,
            dropout_prob=cfg.dropout_prob,
            layer_norm=cfg.layer_norm,
            use_residual=cfg.use_residual,
        )
        # 未来协变量投影层
        self.future_proj = MLPResidual(
            activation_fn=act,
            num_hidden=cfg.future_covariates_num_hidden,
            num_output=cfg.future_covariates_dim,
            dropout_prob=cfg.dropout_prob,
            layer_norm=cfg.layer_norm,
            use_residual=cfg.use_residual,
        )
        # 第一组编码器（堆叠MLP残差块）
        self.encoder1 = StackedMLPResidual(
            activation_fn=act,
            num_hiddens=cfg.encoder_decoder_num_hiddens,
            dropout_prob=cfg.dropout_prob,
            layer_norm=cfg.layer_norm,
            use_residual=cfg.use_residual,
        )
        # 第二组编码器，输出维度扩展以适应预测步长
        second_dims = tuple(list(cfg.encoder_decoder_num_hiddens[:-1]) + [cfg.decoder_dim * cfg.pred_len])
        self.encoder2 = StackedMLPResidual(
            activation_fn=act,
            num_hiddens=second_dims,
            dropout_prob=cfg.dropout_prob,
            layer_norm=cfg.layer_norm,
            use_residual=cfg.use_residual,
        )
        # 时序解码MLP残差层，输出单步预测
        self.temporal_decoder = MLPResidual(
            activation_fn=act,
            num_hidden=cfg.temporal_decoder_num_hidden,
            num_output=1,
            dropout_prob=cfg.dropout_prob,
            layer_norm=cfg.layer_norm,
            use_residual=cfg.use_residual,
        )
        # 残差线性映射层，将过去时序映射到预测步长
        self.res_dense = nn.Dense(features=cfg.pred_len)

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,            # [B, C, F_neuron]
        static_cov: jnp.ndarray,   # [F_neuron, STATIC_DIM] or [B, F_neuron, STATIC_DIM]
        past_cov: jnp.ndarray,     # [B, C, COV_DIM]
        future_cov: jnp.ndarray,   # [B, Tp, COV_DIM]
        train: bool = False,
    ) -> jnp.ndarray:              # returns [B, Tp, F_neuron]
        """
        模型前向推理主函数。
        参数：
            x: 过去神经元时序活动
            static_cov: 神经元静态协变量
            past_cov  : 过去刺激协变量
            future_cov: 未来刺激协变量
            train: 是否训练模式(控制dropout等)
        返回：
            预测神经元活动
        """
        cfg = self.config
        B, C, F_neuron = x.shape
        Tp = cfg.pred_len

        # 可逆实例归一化
        if cfg.instance_norm:
            rev_in = ReversibleInstanceNorm()
            x, stats = rev_in(x)
        # 展平过去时序 → (B*F, C)
        past_ts = x.transpose(0, 2, 1).reshape(-1, C)
        if cfg.ablate_past_timeseries:
            past_ts = jnp.zeros_like(past_ts)
        # 展平静态协变量 → (B*F, A)
        if static_cov.ndim == 2:
            static = jnp.broadcast_to(static_cov[None], (B, F_neuron, static_cov.shape[1]))
        else:
            static = static_cov
        static_flat = static.reshape(-1, static.shape[-1])
        if cfg.ablate_static_covariates:
            static_flat = jnp.zeros_like(static_flat)
        # 过去协变量重复并投影
        pc = jnp.repeat(past_cov[:, None, :, :], F_neuron, axis=1).reshape(-1, C, COV_DIM)
        pc_proj = self.past_proj(pc, train)
        if cfg.ablate_past_covariates:
            pc_proj = jnp.zeros_like(pc_proj)
        # 未来协变量重复并投影 
        fc = jnp.repeat(future_cov[:, None, :, :], F_neuron, axis=1).reshape(-1, Tp, COV_DIM)
        fc_proj = self.future_proj(fc, train)
        if cfg.ablate_future_covariates:
            fc_proj = jnp.zeros_like(fc_proj)
        # 特征拼接
        feat = jnp.concatenate([
            past_ts,
            static_flat,
            pc_proj.reshape(-1, C * cfg.past_covariates_dim),
            fc_proj.reshape(-1, Tp * cfg.future_covariates_dim),
        ], axis=-1)
        # 编码层
        h1 = self.encoder1(feat, train)
        h2 = self.encoder2(h1, train).reshape(-1, Tp, cfg.decoder_dim)
        # 时序解码与残差连接
        merged = jnp.concatenate([h2, fc_proj], axis=-1)
        out_reg = self.temporal_decoder(merged.reshape(-1, merged.shape[-1]), train).reshape(-1, Tp)
        res = self.res_dense(past_ts)
        out = (out_reg + res).reshape(B, F_neuron, Tp).transpose(0, 2, 1)
        # 反归一化
        if cfg.instance_norm and cfg.revert_instance_norm:
            out = rev_in.revert(out, stats)

        return out


# —— 工厂函数 —— #
def build_tide_model(
    context_len: int,
    pred_len:    int,
    seed:        int,
    effective_F: int,
) -> Tuple[nn.Module, Any]:
    """
    构建并初始化 TiDE 时序预测模型。
    参数:
      context_len: 输入上下文长度 C。
      pred_len:    预测步长 H。
      seed:        随机种子。
      effective_F: 神经元数。
    返回:
      model:       TiDE模型实例。
      init_params: 模型初始化参数PyTree。
    """
    cfg = TideConfig(pred_len=pred_len)
    model = Tide(config=cfg)
    rng = jax.random.PRNGKey(seed)

    dummy_x  = jnp.zeros((1, context_len, effective_F), dtype=jnp.float32)
    dummy_sc = jnp.zeros(((effective_F, STATIC_DIM)),     dtype=jnp.float32)
    dummy_pc = jnp.zeros((1, context_len, COV_DIM), dtype=jnp.float32)
    dummy_fc = jnp.zeros((1, pred_len,   COV_DIM),   dtype=jnp.float32)

    variables = model.init(rng, dummy_x, dummy_sc, dummy_pc, dummy_fc, train=False)
    return model, variables["params"]
