# ZAPBench 全脑活动预测复现

本项目基于 [ZAPBench](https://google-research.github.io/zapbench) 论文，实现了时序模型和体积 U-Net 的训练、验证、测试。

## 目录结构

```plaintext
.
├── config.py             # 全局配置：数据路径、超参、模型选项
├── dataset_loader.py     # 流式加载脚本：trace/stimuli/static
├── utils.py              # 通用工具：存档、分割、TensorStore
├── train_utils.py        # 时序模型训练与验证主流程
├── test_utils.py         # 时序模型测试评估脚本
├── train_unet.py         # U-Net 训练主流程
├── test_unet.py          # U-Net 测评脚本
├── models/               # 各类模型实现
│   ├── __init__.py       # 模型工厂及注册表
│   ├── linear_model.py   # 线性模型
│   ├── tide_model.py     # TiDE 模型
│   ├── tsmix_model.py    # TSMixer & TimeMix
│   ├── unet_model.py     # U-Net 模型
│   └── util.py           # 模型通用工具
├── main.py               # 主入口，根据 config 调度训练/评测
├── requirements.txt      # Python 依赖列表
├── .gitignore            # Git 忽略配置
└── README.md             # 本文档
```

## 环境依赖

建议使用虚拟环境来隔离依赖：

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 快速开始

1. **训练模型**（默认在 `config.py` 中设置模型）
   ```
   python main.py
   ```

2. **仅测试已保存模型**（在 `config.py` 中设置 `ONLY_TEST = True`）
   ```
   python main.py
   ```


## 输出结果示例
训练 & 测试完成后，会在当前目录生成类似 linear_C_subset_seed_output.json 的结果文件。

   ```
{
  "gain": {
    "h=1": 0.0198786,
    "h=4": 0.0232666,
  ...
  },
  ...
}
   ```
顶层 key：刺激条件名称

二级 key：预测步长 h={step}

value：该条件在该步长上的平均 MAE
