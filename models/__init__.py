# models/__init__.py
"""
模型工厂与注册表
---------------------
    - 统一管理各类模型的注册与构建(Linear/TiDE/TSMixer/TimeMix/UNet等)
    - 提供 get_model_by_name 工厂函数，根据模型名和参数自动返回所需模型及其初始参数
"""


from typing import Callable, Tuple, Any, Optional

import flax.linen as nn

from .linear_model import build_linear_model
from .tide_model import build_tide_model
from .tsmix_model import build_tsmixer_model, build_time_mix_model 
from .unet_model    import build_unet_model

# 如果后续有其他模型，请在此导入并注册

# ---------------- 注册表 ---------------- #
# key: model_name 小写；value: 对应的 build_xxx_model 函数
_MODEL_REGISTRY: dict[str, Callable[..., Tuple[nn.Module, Any]]] = {
    "linear": build_linear_model,
    "tide":   build_tide_model,
    "tsmixer": build_tsmixer_model,
    "time_mix": build_time_mix_model,
    "unet":    build_unet_model,
}

# ---------------- 工厂函数 --------------- #
def get_model_by_name(
    name: str,
    context_len: int,
    pred_len: int,
    *,
    seed: int,
    effective_F: Optional[int] = None,
    **kwargs: Any,  # 支持任意额外关键字参数传递
) -> Tuple[nn.Module, Any]:
    """
    按模型名和参数构建模型实例与初始参数，支持所有主流程变量及未来模型扩展参数。
    """
    key = name.lower()
    if key not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}"
        )
    
    builder = _MODEL_REGISTRY[key]
    
    if key == "unet":
        return builder(
            context_len=context_len,
            pred_len=pred_len,
            seed=seed,
            **kwargs,
        )
    
    return builder(
        context_len=context_len,
        pred_len=pred_len,
        seed=seed,
        effective_F=effective_F,
        **kwargs,
    )
