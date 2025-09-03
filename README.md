# 第五届计图人工智能挑战赛: 人体骨骼生成赛题（赛题二） - 闪电松鼠队方案(B榜第二)

主要方法流程图
<img width="2066" height="3840" alt="Untitled diagram _ Mermaid Chart-2025-09-02-172445" src="https://github.com/user-attachments/assets/7a82dd8d-2c01-49dc-9388-0a89c50bf127" />


## 简介
本项目是第二届计图挑战赛“人体姿态与蒙皮权重生成”赛题的B榜第二名解决方案。我们针对骨骼节点空间位置预测和蒙皮权重预测两个核心任务，设计了相应的模型和训练策略。

- **骨骼节点预测：** 我们在 `PCT` 的基础上，融合了 `PointNet++` 的多尺度分层抽象思想 (`MultiScaleSA`) 和 `Transformer` (`Point_Transformer_Block`) 来更有效地提取局部及全局特征。此外，我们设计了一个基于锚点的关节预测头 (`JointPredictionHead`)，通过加权平均多个锚点预测的相对偏移来得到最终关节坐标，提升了预测的鲁棒性和精度。
- **蒙皮权重预测：** 为了减小训练与测试的域分布差异，我们使用上一步预测出的骨骼节点来训练蒙皮模型。我们为每个 (顶点, 关节) 对构建了包含位置、距离、全局上下文等多维度的特征向量，并使用一个简单的 MLP 直接预测其亲和度，取得了良好的效果。

## 环境配置

本项目在以下环境测试通过：
- **操作系统:** Ubuntu 20.04.3 LTS
- **Python 版本:** 3.9
- **CUDA 版本:** 12.2
- **CUDNN 版本:** 8

我们提供两种方式来配置运行环境，请根据您的实际情况选择其一：

#### 方案一：使用 `environment.yaml` (推荐)
```bash
# 从 environment.yaml 文件创建 conda 环境
conda env create -f environment.yaml

# 激活环境
conda activate jittor_comp_human
```

#### 方案二：使用 `requirements.txt`
```bash
# 创建一个新的 conda 环境
conda create -n jittor_comp_human python=3.9
conda activate jittor_comp_human

# 安装不高于10版本的gcc/g++ (Jittor编译需要)
conda install -c conda-forge gcc=10 gxx=10

# 使用pip安装依赖
pip install -r requirements.txt
```

```

## 运行步骤

#### 1. 数据准备

请将官方提供的数据集解压后放置于 `data/` 目录下，目录结构应如下所示：
```
data/
├── test/
└── train/
```
项目代码结构说明：
```

├── checkpoints/      # 预训练的骨骼和蒙皮权重
├── code/
│   ├── dataset/      # 数据加载Dataset
│   ├── PCT/          # PCT模型源码
│   ├── models/       # 核心模型代码
│   ├── predict_skeleton.py      # 骨骼节点预测脚本
│   ├── predict_skin.py          # 蒙皮权重预测脚本
│   ├── train_skeleton.py        # 骨骼模型训练脚本
│   └── train_skin.py            # 蒙皮模型训练脚本
├── jittor_comp_human/             # 预配置的Conda环境 (备选)
├── requirements.txt
└── environment.yaml
```

#### 2. 推理

所有推理参数都已设置为默认值，使用我们提供的最佳权重直接运行即可。

```bash
# 1. 预测骨骼节点
# 该脚本会加载 checkpoints/skeleton_pkl/ 目录下4个不同epoch的权重进行集成预测
python code/predict_skeleton.py

# 2. 预测蒙皮权重
# 该脚本会加载 checkpoints/skin_pkl/ 目录下3个loss最低的权重进行集成预测
python code/predict_skin.py
```

#### 3. 训练

**注意：** 骨骼模型使用8卡分布式训练，需要预先安装 `OpenMPI`。
```bash
# 安装 OpenMPI (针对Ubuntu系统)
# 参考: https://cg.cs.tsinghua.edu.cn/jittor/tutorial/2020-5-2-16-44-distributed/
sudo apt install openmpi-bin openmpi-common libopenmpi-dev```

训练脚本如下：
```bash
# 使用4卡分布式训练骨骼模型 800 epoch
# (我们在8卡A100-40G上完成训练，总batch_size为480)
mpirun -np 4 python code/train_skeleton.py

# 使用单卡训练蒙皮模型 400 epoch
python code/train_skin.py
```
训练过程中生成的权重和日志将保存在 `output/` 目录下。

## Checkpoint 说明

我们提供的预训练权重位于 `checkpoints/` 目录下：
- **`checkpoints/skeleton_pkl/`**: 包含骨骼模型在800轮训练中，最后4个epoch (796-799) 保存的模型权重。
- **`checkpoints/skin_pkl/`**: 包含蒙皮模型在400轮训练中，验证集 `skin_loss` 最低的3个模型权重。

推理脚本会自动加载这些权重并进行加权平均集成，以获得更稳定和精确的预测结果。

## 线上结果
- **线上结果:** 最终B榜排名第二
- <img width="226" height="140" alt="图片" src="https://github.com/user-attachments/assets/74d105ce-7be0-4268-b831-794cd539bd5e" />


