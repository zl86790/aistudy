# 第二周作业总结报告

本项目包含第二周的四个作业任务，涵盖以下内容：

- 文本分类  
- 问答任务微调  
- Whisper 模型在中法语语音识别任务上的微调与评估

具体文件说明如下：
##### 1、使用完整的 YelpReviewFull 数据集训练，对比看 Acc 最高能到多少。课程代码（ https://github.com/DjangoPeng/LLM-quickstart/blob/main/transformers/fine-tune-quickstart.ipynb ）
- `06_fine-tune-quickstart-all-datasets.ipynb`  
  使用完整数据集进行训练，在 **NVIDIA Quadro RTX 6000（24GB 显存）** 上运行至第 3 个 Epoch，耗时约 **18 小时**。
  
##### 2、加载本地保存的模型，进行评估和再训练更高的 F1 Score。课程代码（ https://github.com/DjangoPeng/LLM-quickstart/blob/main/transformers/fine-tune-QA.ipynb ） 
- `07_fine-tune-QA-overfitting.ipynb`  
  针对问答任务尝试进一步提升 F1 分数，但出现了明显的**过拟合现象**。
- `09_fine-tune-QA-v2-F1-92.45.ipynb`  
  通过进一步的优化将 F1 分数从 **85.19** 提高到了 **92.45**
- `10_fine-tune-QA-retrain-F1-92.62.ipynb`  
  通过进一步的优化将 F1 分数从 **92.45** 提高到了 **92.62**

##### 1、在“LoRA 低秩适配 OpenAI Whisper-Large-V2 语音识别任务”中，为中文语料的训练过程增加过程评估，观察 Train Loss 和 Validation Loss 变化。课程代码（ https://github.com/DjangoPeng/LLM-quickstart/blob/main/peft/peft_lora_whisper-large-v2.ipynb ） 
##### 2、在“LoRA 低秩适配 OpenAI Whisper-Large-V2 语音识别任务”中，当 LoRA 模型训练完成后，使用测试集进行完整的模型评估。课程代码（ https://github.com/DjangoPeng/LLM-quickstart/blob/main/peft/peft_lora_whisper-large-v2.ipynb ） 
- `05_peft_lora_whisper-large-v2-part640.ipynb`  
  中文语音识别任务微调结果（部分数据集）。
- `11_peft_lora_whisper-large-v2-all.ipynb`  
  中文语音识别任务微调结果（完整数据集）。
- `05_peft_lora_whisper-large-v2-fr-jiwer.ipynb`  
  法语语音识别任务微调结果。
- `08_peft_lora_whisper-large-v2-ja-jiwer.ipynb`  
  日语语音识别任务微调结果。


> ⚠️ 从 `datasets v2.14.0`（2024 年初）开始，Hugging Face 官方逐步弃用基于 `.py` 脚本定义的数据集格式。  
> 所以这里采用的是 `mozilla-foundation/common_voice_13_0` 的 **Parquet 格式预处理数据集**。


---

## 📚 作业 1：YelpReviewFull 数据集文本分类任务

- 📁 文件：`01_fine-tune-quickstart`
- 📚 课程链接：[fine-tune-quickstart.ipynb](https://github.com/DjangoPeng/LLM-quickstart/blob/main/transformers/fine-tune-quickstart.ipynb)
- ⚙️ 使用数据：YelpReviewFull 全量数据（实际仅使用了1000条样本，因硬件限制）
- 🎯 任务目标：比较使用少量样本训练时的准确率表现

### 📊 对比结果

| Epoch | Training Loss | Validation Loss | Accuracy |
|-------|---------------|------------------|----------|
| **老师结果** ||||
| 1     | 1.2421        | 1.0909           | 0.526    |
| 2     | 0.9014        | 0.9601           | 0.591    |
| 3     | 0.6382        | 0.9784           | 0.592    |
| **我的结果** ||||
| 1     | 1.2233        | 1.0545           | 0.546    |
| 2     | 0.9227        | 0.9809           | 0.584    |
| 3     | 0.6569        | 0.9997           | 0.609    |

---

## 📚 作业 2：问答系统微调与再训练

- 📁 文件：`02_fine-tune-QA`
- 📚 课程链接：[fine-tune-QA.ipynb](https://github.com/DjangoPeng/LLM-quickstart/blob/main/transformers/fine-tune-QA.ipynb)
- ⚙️ 使用优化设置：使用 16-bit 精度 + 减小 batch size 以适应显存
- 🎯 任务目标：加载本地模型进行评估并优化 F1 分数

### 📊 对比结果

| 指标             | 老师结果     | 我的结果     |
|------------------|--------------|--------------|
| F1 分数          | 83.64        | **85.19**    |
| 精确匹配率 (EM)  | 74.88        | **77.03**    |

---

## 🔊 作业 3：中文 Whisper 模型训练过程分析（LoRA）

- 📁 文件：`03_peft_lora_whisper-large-v2-part640`
- 📚 课程链接：[peft_lora_whisper-large-v2.ipynb](https://github.com/DjangoPeng/LLM-quickstart/blob/main/peft/peft_lora_whisper-large-v2.ipynb)
- 🎯 任务目标：观察训练过程中 Train Loss 和 Validation Loss 的变化

### 📊 对比结果

| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 老师结果 | 1.5024        | 1.0813          |
| 我的结果 | **0.4950**    | **0.3976**      |

> 注：使用老师原始的数据量进行训练，未使用全量语料。

- 📁 文件：`03_peft_lora_whisper-large-v2-part640`
- 📚 课程链接：[peft_lora_whisper-large-v2.ipynb](https://github.com/DjangoPeng/LLM-quickstart/blob/main/peft/peft_lora_whisper-large-v2.ipynb)
- 🎯 任务目标：观察训练过程中 Train Loss 和 Validation Loss 的变化

#### 使用完整中文数据集训练的结果对比

### 📊 对比结果

| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 老师结果 | 1.5024        | 1.0813          |
| 我的结果 | **0.3057**    | **0.361165**      |



---

## 🔊 作业 4：法语 Whisper 模型评估（LoRA）

- 📁 文件：
  - `04_peft_lora_whisper-large-v2-fr-tested`
  - `05_peft_lora_whisper-large-v2-fr-jiwer`
- 📚 课程链接：[peft_lora_whisper-large-v2.ipynb](https://github.com/DjangoPeng/LLM-quickstart/blob/main/peft/peft_lora_whisper-large-v2.ipynb)
- 🎯 任务目标：切换为法语语料，训练并评估 WER / CER

### 📊 测试集评估结果：

- **词错误率（WER）**：0.2532  
- **字符错误率（CER）**：0.0685

### 🎧 随机样本对比：

- **示例 1**  
  - 🟢 真实文本：Elle est aussi alimentée par la décharge du lac Kawacekamik et quelques ruisseaux riverains.  
  - 🔵 预测文本：Elle est aussi alimentée par la décharge du lac Kawa-Sékamik et quelques ruisseaux riverains.

- **示例 2**  
  - 🟢 真实文本：Aucune de celles-là.  
  - 🔵 预测文本：aucune ne celle.

- **示例 3**  
  - 🟢 真实文本：Dès lors l'importance du village grandit avec sa population.  
  - 🔵 预测文本：Dès lors, l'importance du village grandit avec sa cocculation.

- **示例 4**  
  - 🟢 真实文本：Le combien sommes-nous ?  
  - 🔵 预测文本：Le combien sont-nous?

- **示例 5**  
  - 🟢 真实文本：D'étranges contusions sur le cou ne sont pas remarquées par le médecin.  
  - 🔵 预测文本：Détrange contusion sur le cou ne sont pas remarquées par le médecin.




# 补充内容
### 下面内容使用了全量数据重新进行了训练
### 📚 作业 1：YelpReviewFull 数据集文本分类任务
### 使用全量 YelpReviewFull 数据集训练分类模型
📁 文件：06_fine-tune-quickstart-all-datasets

🎯 任务目标：比较使用全量数据集训练后的模型效果，与使用1000条样本时的结果进行对比

### 📊 对比结果

| Epoch | Training Loss | Validation Loss | Accuracy |
|-------|---------------|------------------|----------|
| **老师结果** ||||
| 1     | 1.2421        | 1.0909           | 0.526    |
| 2     | 0.9014        | 0.9601           | 0.591    |
| 3     | 0.6382        | 0.9784           | 0.592    |
| **我的结果** ||||
| 1     | 1.2233        | 1.0545           | 0.546    |
| 2     | 0.9227        | 0.9809           | 0.584    |
| 3     | 0.6569        | 0.9997           | 0.609    |
| **我的结果（全量数据）** ||||
| 1     | 0.8003        | 0.7557           | 0.6711    |
| 2     | 0.6961        | 0.7271           | 0.6838    |
| 3     | 0.5857        | 0.7338           | 0.6935    |
			

✅ 结论：使用全量数据后，模型的准确率显著提升，验证了数据量对模型性能的重要影响。


### 📚 作业 2：问答系统微调与再训练
### 增加了训练轮次
📁 文件：07_fine-tune-QA-overfitting
| Epoch | Training Loss | Validation Loss |
|-------|---------------|------------------|
| 1     | 0.472900        | 1.517271           | 
| 2     | 0.387000        | 1.658875           | 
| 3     | 0.261200        | 1.858123           | 

可以看到随着 训练集Loss越来越低，验证集上Loss反而变得越来越高了，这是典型的过拟合现象
最终F1 Score也下降了
评估结果:
F1分数: 84.09
精确匹配率: 75.16

## 🔊 作业 4：日语 Whisper 模型评估（LoRA）

- 📁 文件：
  - `08_peft_lora_whisper-large-v2-ja-jiwer`
- 📚 课程链接：[peft_lora_whisper-large-v2.ipynb](https://github.com/DjangoPeng/LLM-quickstart/blob/main/peft/peft_lora_whisper-large-v2.ipynb)
- 🎯 任务目标：切换为日语语料，训练并评估 WER / CER

### 代码中使用的内容为 2025年7月30日的NHK对俄国勘察加地震的报道
#### 识别出的语音内容中其实存在一定的错误：
#### 2008時半後でロシアのカムチャツカ半島付近 实际内容为 午前8時半頃...
#### 因为两句的发音有些类似，录音也不是很清楚，所以出现了识别错误
#### 2008時半後	にせんはちじはんご	❌ 错误表达
#### 午前8時半頃	ごぜんはちじはんごろ	✅ 自然表达

### 📊 测试集评估结果：

- **词错误率（WER）**：2.6000  
- **字符错误率（CER）**：1.5376

### 🎧 随机样本对比：

- **示例 1**  
  - 🟢 真实文本：而して斯く我々が何処までも
  - 🔵 预测文本：しかしてかく、われわれがどこまでも。

- **示例 2**  
  - 🟢 真实文本：部屋に箱が六つ置いてあります。
  - 🔵 预测文本：へやにはこがむつおえてあります。

- **示例 3**  
  - 🟢 真实文本：今晩友達がうちに泊まります。
  - 🔵 预测文本：こんばんトモタチがうちにとまります。

- **示例 4**  
  - 🟢 真实文本：本能による適応は直接的である。
  - 🔵 预测文本：本能による適応は直接的である。

- **示例 5**  
  - 🟢 真实文本：そっと階段をのぼった。
  - 🔵 预测文本：そっと階段を登った。

 ### ✅ 结论：
 #### 这里的日语模型输出之所以看起来有很大不同是因为模型的输出中没有使用对应的汉字，而是直接输出了假名

### 而して斯く我々が何処までも
| 编号 | 单词・短语   | 读音（假名） |
|------|--------------|----------------|
| 1    | 而して書く    | しかしてかく     |
| 2    | 我々          | われわれ         |
| 3    | 何処          | どこ             |

### 部屋に箱が六つ置いてあります。
| 编号 | 单词・短语 | 读音（假名） |
|------|------------|----------------|
| 1    | 部屋        | へや             |
| 2    | 箱          | はこ             |
| 3    | 六          | むっつ           |
| 4    | 置いて      | おいて           |

### 今晩友達がうちに泊まります。
| 编号 | 单词・短语   | 读音（假名） |
|------|--------------|----------------|
| 1    | 今晩          | こんばん         |
| 2    | 友達          | ともだち         |
| 3    | 泊まります    | とまります       |



这里确实能看到一些错误，但是总体上没有什么明显的问题。



### 📚 作业 2：问答系统微调与再训练
### 对代码进行了优化，将 F1 分数从 **85.19** 提高到了 **92.62**
📁 文件：09_fine-tune-QA-v2-F1-92.45.ipynb

#### 通过增加 callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] 在验证集loss没有明显变化的时候终止训练
#### 降低学习率到 learning_rate=5e-6
#### 增加评估的次数，每 500或者200个step的时候就进行一次评估 eval_steps=500
#### 每次重新训练时候选择最佳模型进行加载 load_best_model_at_end
#### 基于更大性能更好的模型进行训练 bert-large-uncased-whole-word-masking-finetuned-squad

评估结果:
F1分数: 92.45
精确匹配率: 86.11

📁 文件：10_fine-tune-QA-retrain-F1-92.62.ipynb
#### 降低学习率到 learning_rate=5e-7
#### 增加评估的次数，每 100个step的时候就进行一次评估
评估结果:
F1分数: 92.62
精确匹配率: 86.41
