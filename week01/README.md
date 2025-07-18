# 🧪 模型压缩实验笔记说明

本项目包含两类 Jupyter Notebook 文件：

---

## 📁 original_models

该文件夹内保存的是课程老师提供的 **原始 Notebook 文件**，用于演示自然语言处理、多模态任务等模型的基础使用方法。

---

## 📁 changed_models_small

该文件夹是我在实验过程中，根据 **算力和网络限制**（如带宽不足、显存较小）对模型进行调整后的版本。

为保障任务顺利完成，我尽量选择了 **体积更小、推理更快** 的替代模型，具体如下：

| 任务类型                 | 替代模型名称                                                         |
|--------------------------|----------------------------------------------------------------------|
| ✅ 情感分析               | `uer/roberta-base-finetuned-jd-binary-chinese`                      |
| ✅ 命名实体识别（NER）    | `dslim/bert-base-NER`                                               |
| ✅ 问答系统（QA）         | `deepset/roberta-base-squad2`                                       |
| ✅ 文本摘要               | `sshleifer/distilbart-cnn-12-6`                                     |
| ✅ 音频分类               | `MIT/ast-finetuned-audioset-10-10-0.4593`                           |
| ✅ 自动语音识别（ASR）    | `openai/whisper-base`                                               |
| ✅ 图像分类（CV）         | `Anwarkh1/Skin_Cancer-Image_Classification`                         |
| ✅ 目标检测               | `yainage90/fashion-object-detection`                                |

---

### 🎯 实验目标

- 能在普通 PC、轻量 GPU、或网络不稳定的环境下运行各类深度学习任务
- 保证实验功能不变的同时，尽可能减小模型体积与运行资源消耗