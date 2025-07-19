# 🧪 模型压缩实验笔记说明

本项目包含两个 Jupyter Notebook 文件，分别记录了原始实验内容与模型压缩优化版本：

---

## 📄 original_models.ipynb

该 Notebook 是课程老师提供的**原始实验文件**，用于演示自然语言处理、多模态任务等模型的标准使用方式，所用模型多为默认配置或标准大小。

在本地电脑上尝试构建了环境，然后直接运行了一次。

---

## 📄 changed_models_small.ipynb

该 Notebook 是我在实验过程中，根据**带宽与显存限制**对模型进行简化优化的版本。

为保障任务顺利完成，我尽量选择了 **体积更小、推理更快** 的替代模型，具体如下：

| 任务类型                 | 替代模型名称                                                         |
|--------------------------|----------------------------------------------------------------------|
| ✅ 情感分析               | `uer/roberta-base-finetuned-jd-binary-chinese`                      |
| ✅ NER                   | `dslim/bert-base-NER`                                               |
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

---



## 一、情感分析（Sentiment Analysis）

| 对比维度     | 原始模型：`distilbert-base-uncased-finetuned-sst-2-english` | 修改后模型：`uer/roberta-base-finetuned-jd-binary-chinese` |
|--------------|-------------------------------------------------------------|--------------------------------------------------------------|
| 核心特点     | 英文情感分析专用，对英文识别精准                           | 中文京东评论微调，优化中文语境适配                          |
| 中文处理     | 支持差，易误判（如中文表扬句可能被错分）                   | 有基础识别能力：<br>• "今儿上海可真冷啊" → `positive (0.9026)`<br>• "味道一般" → `positive (0.5883)`（精准度待提升） |
| 英文处理     | 表现优，可准确识别英文正面情感                             | 适配差，英文表扬句被误判为 `negative (0.6446)`              |

---

## 二、命名实体识别（NER）

| 对比维度     | 原始模型：`dbmdz/bert-large-cased-finetuned-conll03-english` | 修改后模型：`dslim/bert-base-NER` |
|--------------|---------------------------------------------------------------|------------------------------------|
| 核心特点     | 大模型，实体识别覆盖广                                       | 轻量模型，聚焦基础实体类型        |
| 识别效果     | 实体得分稍高（如 ORG ≈ 0.9675）                              | 精准识别边界与类型：<br>• `ORG`: Hugging Face → `0.9287`<br>• `MISC`: French → `0.9996`<br>• `LOC`: New York City → `0.9995` |

---

## 三、问答任务（Question Answering）

| 对比维度     | 原始模型：`distilbert-base-cased-distilled-squad` | 修改后模型：`deepset/roberta-base-squad2` |
|--------------|---------------------------------------------------|--------------------------------------------|
| 核心特点     | 轻量模型，适合快速推理                           | SQuAD2.0 微调，优化长文本问答              |
| 回答准确率   | 简单问题得分略低，复杂问题易偏移                 | 表现稳定：<br>• "中国的首都是？" → `Beijing (0.7504)`<br>• "仓库名称？" → 精准识别 `score: 0.9068` |

---

## 四、摘要任务（Summarization）

| 对比维度     | 原始模型：`facebook/bart-large-cnn` | 修改后模型：`sshleifer/distilbart-cnn-12-6` |
|--------------|-------------------------------------|---------------------------------------------|
| 核心特点     | 完整版 BART，适合长文本，信息细致   | 蒸馏版 BART，速度更快，聚焦关键信息        |
| 摘要效果     | 超长摘要可能超限（>32词），略显冗余 | 控长简洁，8-32 词提炼 Transformer 核心思想 |

---

## 五、音频处理任务

### 5.1 音频分类

| 对比维度     | 原始模型：`superb/wav2vec2-base-superb-ks` | 修改后模型：`MIT/ast-finetuned-audioset-10-10-0.4593` |
|--------------|--------------------------------------------|--------------------------------------------------------|
| 核心特点     | 聚焦语音/非语音分类                         | 支持更广泛的音频场景标签（如 Speech、Rain）           |
| 分类结果     | 通常识别为“Speech”“Human Voice”            | 示例结果：<br>• `Speech: 0.4208`<br>• 背景有雨声 → `Rain` 标签识别成功 |

### 5.2 自动语音识别（ASR）

| 对比维度     | 原始模型：`facebook/wav2vec2-large-960h` | 修改后模型：`openai/whisper-base` |
|--------------|-------------------------------------------|------------------------------------|
| 核心特点     | 英文专用，长句断句或漏词较多              | 多语言支持，识别完整，标点自然    |
| 识别效果     | 英文演讲常漏标点                          | 示例：`"I have a dream..."` → 完整转写 ✅ |

---

## 六、图像分类（Image Classification）

| 对比维度     | 原始模型：`google/vit-base-patch16-224` | 修改后模型：`Anwarkh1/Skin_Cancer-Image_Classification` |
|--------------|------------------------------------------|----------------------------------------------------------|
| 核心特点     | 通用视觉模型，支持广泛类别               | 专用于皮肤癌检测，任务适配窄                          |
| 分类结果     | 准确识别“猫”“熊猫”（score > 0.9）       | 不适配通用图像，误判严重：<br>• 猫图片 → `melanoma (0.5087)`<br>• 熊猫图片 → `benign_keratosis (0.4991)` |

---

## 七、目标检测（Fashion Object Detection）

| 对比维度     | 原始模型：`facebook/detr-resnet-50` | 修改后模型：`yainage90/fashion-object-detection` |
|--------------|------------------------------------------|----------------------------------------------------------|
| 核心特点     | 通用目标检测模型，覆盖 80 + 常见物体（人、车、动物等）              | 专注时尚领域，支持服装、配饰等细分类别（如连衣裙、鞋子、帽子）                          |
| 分类结果     | 准确识别“猫”“狗”       | 不适配通用图像，误判严重：识别成了外套 |

---

## 🧾 总结

> ✅ **原始模型**：适合通用 NLP/CV 任务，泛化能力强，跨语言表现更稳定。  
> 🧠 **修改后模型**：在特定任务上（如中文文本、语音识别）性能更佳，但泛化性较弱，需严格匹配任务类型。 







# changed_models_large 文件使用了尽可能大的模型，期望获取更精准的结果。
# 以下是调整模型后的输出与老师原文件输出的对比

---

# 🔧 使用 Pipeline API 调用更多预定义任务

---

## 🏷️ Named Entity Recognition（实体识别）

模型：[`dslim/bert-large-NER`](https://huggingface.co/dslim/bert-large-NER)
模型大小：**1.33GB**
下载量：**148,469**

输入文本：

> Hugging Face is a French company based in New York City.

**原始输出（未合并）**：

```yaml
{'entity': 'I-ORG', 'score': 0.9968, 'index': 1, 'word': 'Hu', 'start': 0, 'end': 2}
{'entity': 'I-ORG', 'score': 0.9293, 'index': 2, 'word': '##gging', 'start': 2, 'end': 7}
{'entity': 'I-ORG', 'score': 0.9763, 'index': 3, 'word': 'Face', 'start': 8, 'end': 12}
{'entity': 'I-MISC', 'score': 0.9983, 'index': 6, 'word': 'French', 'start': 18, 'end': 24}
{'entity': 'I-LOC', 'score': 0.999, 'index': 10, 'word': 'New', 'start': 42, 'end': 45}
{'entity': 'I-LOC', 'score': 0.9987, 'index': 11, 'word': 'York', 'start': 46, 'end': 50}
{'entity': 'I-LOC', 'score': 0.9992, 'index': 12, 'word': 'City', 'start': 51, 'end': 55}
```

**变更后输出**：

```yaml
{'entity': 'B-ORG', 'score': np.float32(0.995), 'word': 'Hu', 'start': 0, 'end': 2}
{'entity': 'I-ORG', 'score': np.float32(0.9456), 'word': '##gging', 'start': 2, 'end': 7}
{'entity': 'I-ORG', 'score': np.float32(0.9917), 'word': 'Face', 'start': 8, 'end': 12}
{'entity': 'B-MISC', 'score': np.float32(0.9977), 'word': 'French', 'start': 18, 'end': 24}
{'entity': 'B-LOC', 'score': np.float32(0.9986), 'word': 'New', 'start': 42, 'end': 45}
{'entity': 'I-LOC', 'score': np.float32(0.9989), 'word': 'York', 'start': 46, 'end': 50}
{'entity': 'I-LOC', 'score': np.float32(0.9994), 'word': 'City', 'start': 51, 'end': 55}
```

**合并实体结果对比：**

原始：

```yaml
[{'entity_group': 'ORG', 'score': 0.9674639, 'word': 'Hugging Face', 'start': 0, 'end': 12},
 {'entity_group': 'MISC', 'score': 0.99828726, 'word': 'French', 'start': 18, 'end': 24},
 {'entity_group': 'LOC', 'score': 0.99896103, 'word': 'New York City', 'start': 42, 'end': 55}]
```

变更后：

```yaml
[{'entity_group': 'ORG', 'score': np.float32(0.9674638), 'word': 'Hugging Face', 'start': 0, 'end': 12},
 {'entity_group': 'MISC', 'score': np.float32(0.99828726), 'word': 'French', 'start': 18, 'end': 24},
 {'entity_group': 'LOC', 'score': np.float32(0.99896103), 'word': 'New York City', 'start': 42, 'end': 55}]
```

📝 **结论**：更换模型后，NER 任务的精度提升空间有限。

---

## ❓ Question Answering（问答）

模型：[deepset/roberta-large-squad2](https://huggingface.co/deepset/roberta-large-squad2)
模型大小：**1.42GB**
下载量：**30,383**

**Q1: What is the name of the repository?**

```yaml
原始: score: 0.9327, start: 30, end: 54, answer: huggingface/transformers
变更后: score: 0.9855, start: 30, end: 54, answer: huggingface/transformers
```

**Q2: What is the capital of China?**

```yaml
原始: score: 0.9458, start: 115, end: 122, answer: Beijing
变更后: score: 0.8053, start: 115, end: 122, answer: Beijing
```

📝 **结论**：对于简单问答，更换模型带来的精度提升也十分有限。

---

## ✂️ Summarization（文本摘要）

模型：[facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn)
模型大小：**1.63GB**
下载量：**3,138,984**

**示例 1：**

```yaml
原始: the Transformer is the first sequence transduction model based entirely on attention...
变更后: The Transformer is the first sequence transduction model based entirely on attention. It replaces the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention...
```

**示例 2：**

```yaml
原始: large language models (LLMs) are very large deep learning models pre-trained on vast amounts of data...
变更后: Large language models (LLM) are very large deep learning models that are pre-trained on vast amounts of data. Transformer LLMs are capable of unsupervised training...
```

📝 **结论**：摘要任务中模型更换效果明显，内容更完整、语言更自然。

---

## 🔊 Audio 音频分类

模型：[MIT/ast-finetuned-audioset-16-16-0.442](https://huggingface.co/MIT/ast-finetuned-audioset-16-16-0.442)
模型大小：**344MB**
下载量：**188**

**原始分类结果：**

```yaml
[{'score': 0.4532, 'label': 'hap'},
 {'score': 0.3622, 'label': 'sad'},
 {'score': 0.0943, 'label': 'neu'},
 {'score': 0.0903, 'label': 'ang'}]
```

**变更后结果（部分）：**

```yaml
[{'score': 0.4519, 'label': 'Speech'},
 {'score': 0.1678, 'label': 'Male speech, man speaking'},
 {'score': 0.1043, 'label': 'Narration, monologue'},
 {'score': 0.0598, 'label': 'Rain on surface'},
 {'score': 0.0421, 'label': 'Rain'},
 {'score': 0.0279, 'label': 'Raindrop'},
 {'score': 0.0172, 'label': 'Run'},
 ... 共数十类 ...]
```

📝 **结论**：新模型可识别的类别更多，分类更细致，表现显著提升。

---

## 🎙️ Automatic Speech Recognition（ASR）

模型：[openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3)
模型大小：**3.09GB**
下载量：**3,085,363**

**语音输入：**

```yaml
{'text': ' I have a dream that one day this nation will rise up and live out the true meaning of its creed.'}
```

**变更前后结果一致**

📝 **结论**：测试样本简单，未观察到显著差异。

---

## 🖼️ Computer Vision（图像分类）

模型：[google/vit-large-patch16-224](https://huggingface.co/google/vit-large-patch16-224)
模型大小：**1.22GB**
下载量：**27,276**

**示例 1（野猫图像）**

```yaml
原始: top-1 'lynx' score: 0.4335
变更后: top-1 'lynx' score: 0.2391
```

**示例 2（大熊猫图像）**

```yaml
原始: 'giant panda' score: 0.9962
变更后: 'giant panda' score: 0.9747
```

📝 **结论**：将 ViT-Base（320MB）升级为 ViT-Large（1.22GB）后，识别精度未提升，甚至略有下降。
尤其第一张图中的 **Pallas's cat（兔狲）** 未被正确识别，说明通用模型仍有盲区。

---

## 📦 Object Detection（目标检测）

模型：`facebook/detr-resnet-101`
模型大小：**243MB**
下载量：**41,130**

**示例结果：**

```yaml
原始:
  - cat, score: 0.9985, box: [78, 57, 309, 371]
  - dog, score: 0.9890, box: [279, 20, 482, 416]

变更后:
  - dog, score: 0.9987, box: [281, 21, 486, 415]
  - cat, score: 0.9950, box: [76, 63, 314, 373]
```

📝 **结论**：从 ResNet-50（165MB）升级到 ResNet-101（243MB）后，检测框精度略有提升，但由于图像过于简单，变化不显著。

---


