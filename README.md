# VGG-ResNet: CIFAR-10 残差连接对照实验

用同一套 VGG-16 CIFAR 网络比较普通卷积和恒等残差连接。两个模型使用完全相同的卷积层、分类头和可训练参数；残差模型只在输入、输出形状相同的卷积处执行 `输出 = ReLU(卷积输出 + 输入)`，不使用投影层。因此 plain 与 residual 在相同 BN 设置下参数量一致，结构差异只有 shortcut。

## 实验约定

- VGG-16 主干保留 13 个 3×3 卷积层和 5 次最大池化，适配 32×32 图像；使用全局平均池化和 10 类线性分类头。
- 从 CIFAR-10 官方训练集按类别分层、固定 seed 划分 45,000 个训练样本和 5,000 个验证样本，每类验证样本固定 500 个。实际索引保存到 `results/splits/seed_<seed>.json`，后续同 seed 运行复用该划分。验证集只用于选取最佳 checkpoint。
- 训练集使用随机裁剪和随机水平翻转；验证集与测试集只做标准化。
- 测试集在训练结束并载入验证集最优 checkpoint 后才创建并评估，不参与模型选择或调参。
- BN 是可选消融项。比较 shortcut 时要保持 BN 设置相同；比较 BN 时保持模型和其余配置相同。
- seed、数据划分、DataLoader shuffle 和 PyTorch 确定性选项均有固定配置。不同硬件、驱动或 PyTorch/CUDA 版本之间仍可能存在少量数值差异。

## 环境

建议使用 Python 3.10 或更高版本。按本机 CUDA/PyTorch 官方安装说明安装匹配的 `torch` 和 `torchvision`，再安装其余依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

程序默认在 CUDA 可用时使用 GPU，否则使用 CPU。可通过 `--device cpu` 或 `--device cuda` 显式指定。首次训练时，torchvision 会自动下载 CIFAR-10 到 `data/`；项目初始化和代码准备本身不会下载数据。

## 运行实验

在项目根目录执行，先后完成 plain 和 residual 两组：

```bash
python train.py --model plain
python train.py --model residual
```

BN 消融使用相同配置分别运行：

```bash
python train.py --model plain --batch-norm
python train.py --model residual --batch-norm
```

也可以不使用 BN（默认），或覆盖训练轮数、batch size、随机种子：

```bash
python train.py --model residual --epochs 100 --batch-size 128 --seed 2026
```

`config.yaml` 保存共享训练参数。命令行只覆盖本次运行的对应值。若要做多 seed 稳健性对比，可将 plain/residual 成对使用相同的 seed，并分别重复运行。

## 输出

每次运行保存到 `results/<run-name>/`：

- `config.json`：实际使用的配置
- `best.pt`：验证集表现最佳的模型权重及元数据
- `epochs.csv`：逐 epoch 的训练/验证 loss、accuracy 和学习率
- `curves.png`：loss 与 accuracy 曲线
- `summary.json`：该次运行的最佳验证和测试结果

`results/summary.csv` 汇总所有运行，便于比较不同模型和 BN 设置。数据、权重和生成结果均被 `.gitignore` 排除，不会提交到 Git。

## 可复现对照建议

先用相同 seed、优化器、增强、batch size 和训练轮数跑 plain/residual。以验证集最高 accuracy 选择 checkpoint；只在确定配置后查看测试结果。初次学习可先用较少 epoch 检查流程，再用相同完整设置正式比较。报告至少记录模型、BN、seed、参数量、最佳 epoch、最佳验证 accuracy 和测试 accuracy；如需科研结论，建议使用多个配对 seed 报告均值和标准差。

## 参考资料

- [CIFAR-10 官方页面](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch torchvision VGG 实现](https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
