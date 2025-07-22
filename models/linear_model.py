# models/linear_model.py
"""
线性模型结构与工厂函数
---------------------
    - 定义linear线性时序预测模型
    - 提供build_linear_model工厂函数统一模型构建与参数初始化
"""


from flax import linen as nn
from flax import struct
import jax.numpy as jnp
import jax
from typing import Any

from config import NUM_NEURONS


# —— 配置类 —— #
@struct.dataclass
class NlinearConfig:
    """
    线性模型配置类，包含输出步数、初始化方式与归一化开关等参数。
    """
    num_outputs: int              #预测的时间步长数量。
    constant_init: bool = True    #是否使用常数初始化权重。True，权重初始化为常数值 1 / 输入特征数目。
    normalization: bool = False   #是否对输入数据进行增量归一化。False


# —— 主模型类 —— #
class Nlinear(nn.Module):
    """
    线性基线模型，支持可选增量归一化与常数初始化。
    输入 shape: (B, C, F)  输出 shape: (B, H, F)
    """
    config: NlinearConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        del train

        if self.config.normalization:
            last_step = x[:, -1:, :]
            x = x - last_step

        # 转置到 (B, F, C)
        x = jnp.transpose(x, axes=(0, 2, 1))

        dense = nn.Dense(
            features=self.config.num_outputs,
            use_bias=True,
            kernel_init=(
                nn.initializers.constant(1.0 / x.shape[-1])
                if self.config.constant_init
                else nn.initializers.lecun_normal()
            ),
            bias_init=nn.initializers.zeros
        )
        x = dense(x)  # (B, F, H)
        # 转回 (B, H, F)
        x = jnp.transpose(x, axes=(0, 2, 1))

        if self.config.normalization:
            return x + last_step
        else:
            return x


# —— 工厂函数 —— #
def build_linear_model(
    context_len: int,
    pred_len: int,
    effective_F: int,
    seed: int = 0,
    normalization: bool = False,
    constant_init: bool = True,
) -> tuple[Nlinear, Any]:
    """
    构建并初始化 Linear 模型。
    参数:
      context_len:   上下文长度 C
      pred_len:      预测长度 H
      effective_F:   神经数
      normalization: 是否启用增量归一化
      constant_init: 是否使用常数初始化
      seed:          随机种子
    返回:
      model:       Nlinear 实例
      init_params: 初始化参数 PyTree
    """
    model = Nlinear(
        config=NlinearConfig(
            num_outputs=pred_len,
            constant_init=constant_init,
            normalization=normalization
        )
    )
    dummy_input = jnp.zeros((1, context_len, effective_F), dtype=jnp.float32)
    rng = jax.random.PRNGKey(seed)
    variables = model.init(rng, dummy_input)
    return model, variables['params']
