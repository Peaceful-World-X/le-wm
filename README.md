# LeWorldModel

### 基于像素的稳定端到端联合嵌入预测架构

https://github.com/lucas-maes/le-wm

[Lucas Maes\*](https://x.com/lucasmaes_) 、 [Quentin Le Lidec\*](https://quentinll.github.io/) 、 [Damien Scieur](https://scholar.google.com/citations?user=hNscQzgAAAAJ&hl=fr) 、 [Yann LeCun](https://yann.lecun.com/) 和 [Randall Balestriero](https://randallbalestriero.github.io/)

**摘要：** 联合嵌入预测架构（JEPA）为在紧凑的潜在空间中学习世界模型提供了一个引人注目的框架，但现有方法仍然脆弱，依赖于复杂的多项损失函数、指数移动平均、预训练编码器或辅助监督来避免表征崩溃。本文提出了 LeWorldModel（LeWM），这是第一个仅使用两个损失项即可从原始像素稳定地进行端到端训练的 JEPA：一个用于预测下一嵌入的损失函数和一个用于强制潜在嵌入服从高斯分布的正则化项。与目前唯一的端到端替代方案相比，LeWM 将可调损失超参数从六个减少到一个。LeWM 可在单个 GPU 上于数小时内训练约 1500 万个参数，其规划速度比基于基础模型的世界模型快 48 倍，同时在各种 2D 和 3D 控制任务中保持竞争力。除了控制任务之外，我们还证明 LeWM 的潜在空间通过探测物理量编码了有意义的物理结构。意外评估证实，该模型能够可靠地检测出物理上不合理的事件。

**\[ [论文](https://arxiv.org/pdf/2603.19312v1) | [检查点](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e?usp=sharing) | [数据](https://huggingface.co/collections/quentinll/lewm) | [网站](https://le-wm.github.io/) \]**


## 使用代码

该代码库基于 [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) 进行环境管理、规划和评估，并[基于 stable-pretraining](https://github.com/galilai-group/stable-pretraining) 进行训练。它们共同将该代码库的核心贡献简化为：模型架构和训练目标。

**安装：**

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

## 数据

数据集采用 HDF5 格式以实现快速加载。从 [HuggingFace](https://huggingface.co/collections/quentinll/lewm) 下载数据并使用以下命令解压缩：

```bash
tar --zstd -xvf archive.tar.zst
```

将提取的 `.h5` 文件放置在 `$STABLEWM_HOME` 目录下（默认为 `~/.stable-wm/` ）。您可以覆盖此路径：

```bash
export STABLEWM_HOME=/path/to/your/storage
```

数据集名称指定时不带 `.h5` 扩展名。例如， `config/train/data/pusht.yaml` 引用了 `pusht_expert_train` ，它解析为 `$STABLEWM_HOME/pusht_expert_train.h5` 。

## 训练

`jepa.py` 包含 LeWM 的 PyTorch 实现。训练通过 `config/train/` 下的 Hydra 配置文件进行配置。

训练前，请在 `config/train/lewm.yaml` 中设置 WandB `entity` 和 `project` ：

```yaml
wandb:
  config:
    entity: your_entity
    project: your_project
```

启动培训：

```bash
python train.py data=pusht
```

检查点完成后会保存到 `$STABLEWM_HOME` 目录。

有关基线脚本，请参阅 stable-worldmodel [脚本](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train)文件夹。

## 规划

评估配置位于 `config/eval/` 下。将 `policy` 字段设置为**相对于 `$STABLEWM_HOME`** 检查点路径，不带 `_object.ckpt` 后缀：

```bash
# ✓ correct
python eval.py --config-name=pusht.yaml policy=pusht/lewm

# ✗ incorrect
python eval.py --config-name=pusht.yaml policy=pusht/lewm_object.ckpt
```

## 预训练检查点

预训练的检查点可在 [Google 云端硬盘](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e)上找到。下载检查点存档并将解压后的文件放在 `$STABLEWM_HOME/` 目录下。

| 方法 | 两室 | 推 | 立方体 | 取物器 |
| --- | --- | --- | --- | --- |
| pldm | ✓ | ✓ | ✓ | ✓ |
| lejepa | ✓ | ✓ | ✓ | ✓ |
| ivl | ✓ | ✓ | ✓ | — |
| iql | ✓ | ✓ | ✓ | — |
| gcbc | ✓ | ✓ | ✓ | — |
| dinowm | ✓ | ✓ | — | — |
| dinowm_noprop | ✓ | ✓ | ✓ | ✓ |

## 加载检查点

每个 tar 归档文件每个检查点包含两个文件：

*   `<name>_object.ckpt` — 一个序列化的 Python 对象，方便加载； `eval.py` 和 `stable_worldmodel` API 都使用这个对象。
*   `<name>_weight.ckpt` — 仅包含权重的检查点（ `state_dict` ），用于将权重加载到您自己的模型实例中的情况。

要通过 `stable_worldmodel` API 加载对象检查点：

```python
import stable_worldmodel as swm

# Load the cost model (for MPC)
cost = swm.policy.AutoCostModel('pusht/lewm')
```

此函数接受以下参数：

*   `run_name` — **相对于 `$STABLEWM_HOME`** 检查点路径，不包含 `_object.ckpt` 后缀
*   `cache_dir` — 检查点根目录的可选覆盖设置（默认为 `$STABLEWM_HOME` ）

返回的模块处于 `eval` 模式，其 PyTorch 权重可通过 `.state_dict()` 访问。

## 联系方式及投稿

欢迎提交[问题](https://github.com/lucas-maes/le-wm/issues) ！如有任何疑问或合作意向，请联系 `lucas.maes@mila.quebec`