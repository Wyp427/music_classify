# 融合音频特征与歌词文本的多模态音乐风格分类

本项目来源于本科毕业论文，聚焦**多模态音乐风格分类**任务，旨在通过融合音频信号与歌词语义信息，提升复杂场景下的风格识别性能与稳定性。

---

## 1. 项目简介

本项目致力于构建一个融合音频信号与歌词文本信息的多模态音乐风格分类系统。针对传统单一模态模型在复杂音乐场景中表达能力不足的问题，本文提出一种结合深度学习与多模态融合机制的分类框架，通过充分挖掘音频特征与歌词语义之间的互补关系，实现对音乐风格更加精准和鲁棒的识别。

在模型设计上，系统分别对音频信号和歌词文本进行建模，并通过动态融合策略对两种模态信息进行自适应整合，从而提升整体分类性能。同时，针对歌词语义表达的复杂性，引入双层语义融合与辅助监督机制，有效增强文本分支的表达能力。

---

## 2. 项目特点（创新点）

- **多模态融合建模**：联合利用音频信号与歌词文本信息，突破单模态信息瓶颈  
- **双分支结构设计**：构建音频分支 + 歌词分支的协同建模框架  
- **动态融合策略**：根据不同模态的重要性自适应分配权重，提高模型泛化能力  
- **歌词语义增强机制**：
  - 双层语义融合（局部 + 全局语义）
  - 重复感知的自适应融合机制
  - 辅助监督优化策略提升语义表达稳定性
- **可扩展性强**：支持多种音频特征（MFCC / Mel / 混合特征）  
- **工程化实现**：完整训练、测试、推理流程，支持快速部署

---

## 3. 模型结构说明

整体模型采用**双分支 + 融合层**的架构：

### （1）音频分支（Audio Branch）
- **输入**：音频信号（`.wav` 等格式）
- **特征提取**：
  - MFCC
  - Mel Spectrogram
  - 或 MFCC + Mel 融合特征
- **模型结构**：双分支 CNN（支持特征级门控融合）
- **输出**：音频特征向量（高层语义表示）

### （2）歌词分支（Lyrics Branch）
- **输入**：歌词文本
- **文本编码**：基于预训练语言模型（BERT）
- **核心机制**：
  - 双层语义融合（句级 + 篇章级）
  - 重复信息感知机制（增强关键词表达）
  - 自适应语义融合（动态调整语义权重）
- **辅助优化**：引入辅助监督任务，提高语义表示质量
- **输出**：歌词语义向量

### （3）融合策略（Fusion Module）
- **融合方式**：
  - 拼接融合（Concatenation）
  - 决策加权融合（Weighted Fusion）
  - 动态自适应融合（Dynamic Fusion，核心创新）
- **动态融合特点**：
  - 根据输入样本自动调整音频/歌词权重
  - 提升对不同风格音乐的适应能力

### （4）分类层
- 全连接层（FC）
- Softmax 输出风格类别概率

---

## 4. 数据集说明

数据集为自建多模态音乐数据集，包含：

```text
dataset_multy2/
├── blues/
│   ├── music/
│   └── lyric/
├── classical/
│   ├── music/   （无歌词，仅音频）
├── disco/
├── hiphop/
├── jazz/
├── metal/
├── pop/
├── reggae/
├── rock/

数据特点：
- 每个类别包含：
  - 音频文件（`.wav`）
  - 歌词文件（`.txt`）
- `classical` 类别为纯音频数据（无歌词）
- 支持自动对齐音频与歌词

---

## 5. 环境配置（Installation）

> 建议使用 Python 3.9+，并在虚拟环境中安装依赖。

### 5.1 克隆项目

```bash
git clone <your-repo-url>
cd music_classify
```

### 5.2 创建虚拟环境（可选但推荐）

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

### 5.3 安装依赖

项目当前未提供统一的 `requirements.txt`，可先按核心依赖安装：

```bash
pip install torch torchvision torchaudio
pip install transformers librosa soundfile numpy scipy scikit-learn tqdm matplotlib
pip install streamlit flask flask-cors flask-sqlalchemy pymysql requests
```

> 如需 GPU 版本 PyTorch，请根据本机 CUDA 版本到官方安装页替换对应命令。

### 5.4 数据与模型文件准备

- 将处理后的数据放在默认目录：
  - `dataset_multy2_processed/audio`
  - `dataset_multy2_processed/lyrics`
  - `dataset_multy2_processed/metadata/song_mapping.json`
- 训练/推理前请确认以下权重与配置文件可用（位于 `core/`）：
  - `best_model.pth` / `best_model_config.json`（音频）
  - `lyrics_best_model.pth` / `lyrics_best_model_config.json`（歌词）
  - `multimodal_best_model_dynamic.pth`（多模态融合头）

---

## 6. 使用方法（训练 / 测试 / 推理）

> 以下命令默认在项目根目录执行；若路径不同，请按实际情况调整。

### 6.1 训练

#### A. 音频分支训练（3 折交叉验证）

```bash
python core/audio_train.py

> `core/audio_train.py` 中通过 `hyperparameters` 字典配置数据路径、学习率、特征类型、epoch 等参数。

#### B. 歌词分支训练（3 折交叉验证）

```bash
python core/lyrics_train.py
```

> `core/lyrics_train.py` 中通过 `config` 字典配置 BERT、训练超参数、数据路径等。

#### C. 多模态融合训练（支持三种融合策略）

```bash
python core/multimodal_train.py --fusion concat
python core/multimodal_train.py --fusion weighting
python core/multimodal_train.py --fusion dynamic
```

常用可选参数示例：

```bash
python core/multimodal_train.py \
  --fusion dynamic \
  --epochs 20 \
  --batch_size 8 \
  --fusion_lr 1e-4 \
  --audio_lr 5e-5 \
  --lyrics_lr 5e-4 \
  --song_mapping dataset_multy2_processed/metadata/song_mapping.json
```

### 6.2 测试/评估

- 训练后会在 `core/` 下生成测试结果 JSON（例如 `multimodal_test_results_dynamic.json`）。
- 关键指标为 **Macro-F1**，并支持查看各类别 F1、混淆矩阵和融合权重分布。

如需快速进行单文件音频预测测试：

```bash
python core/test.py

> 运行前请先在 `core/test.py` 中修改待测音频路径与模型文件路径。

### 6.3 推理与应用

#### A. Streamlit 可视化演示

```bash
streamlit run core/server.py
```

可用于：
- 上传音频进行风格预测
- 输入/上传歌词进行歌词分支预测
- 联合音频+歌词执行多模态预测并展示各类别概率

#### B. Flask API 服务

```bash
python core/web.py
```

默认启动后端服务（`0.0.0.0:5000`），可用于与前端系统或第三方应用集成。

---

## 7. 项目结构

```text
music_classify/
├── README.md
├── dataset_multy2_processed/
│   └── metadata/
│       └── song_mapping.json
└── core/
    ├── audio_train.py                  # 音频分支训练（3折）
    ├── lyrics_train.py                 # 歌词分支训练（3折）
    ├── multimodal_train.py             # 多模态融合训练（concat/weighting/dynamic）
    ├── test.py                         # 单样本测试脚本
    ├── server.py                       # Streamlit 可视化推理界面
    ├── web.py                          # Flask API 服务
    ├── data_process.py                 # 音频数据处理
    ├── lyrics_data_process.py          # 歌词数据处理
    ├── process_multimodal_dataset.py   # 多模态数据整理与映射
    ├── feature_utils.py                # MFCC/Mel 特征提取
    ├── fusion_models.py                # 融合层定义（含动态融合）
    ├── lyrics_model.py                 # 歌词分支模型（BERT）
    ├── dual_branch_cnn.py              # 音频双分支CNN模型
    ├── model_factory.py                # 模型构建与加载入口
    ├── metrics_utils.py                # 指标计算工具
    ├── label_mapper.py                 # 标签映射
    └── sql/
        └── music_classify.sql          # 数据库结构（用于Web服务）
```

---

## 实验结果（论文核心结论）

- 多模态模型：**0.587**
- 音频模型：**0.490**
- 歌词模型：**0.478**
- 动态融合策略在多数类别上表现更稳定

---

## 技术栈

- Python
- PyTorch
- Transformers（BERT）
- Librosa
- NumPy

---

## 说明

本项目已实现从**音频输入**到**风格预测输出**的完整流程，具备基础应用能力。未来可继续扩展：

- 更大规模多语种歌词数据
- 更强音频预训练模型（如 AST / HTSAT）
- 端到端在线服务与模型压缩部署
