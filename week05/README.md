# LLaMA 2 指令微调实践（基于 Alpaca 风格与 Dolly-15K 数据集）

本周的作业内容为：
使用课程中的数据集和代码，完成 LLaMA 指令微调训练。
学习重点是：
学习如何LLaMA 2 指令微调，怎么以 Alpaca-Style 格式化指令数据。

---

## 项目背景与目标

### 作业要求

我的理解是，老师希望我们基于 `databricks-dolly-15k` 数据集和课程框架，完成 LLaMA 2 模型的指令微调，核心目标：

- 掌握 LLaMA 2 指令微调的完整流程
- 理解并实践 Alpaca-Style 指令数据的格式化逻辑
- 探索量化与高效训练技术在微调中的应用

---

## 学习过程

### 1. 环境与工具准备

配置训练所需核心环境，包括：

- `torch`  
- `transformers`  
- `datasets`  
- `bitsandbytes`（量化）  
- `peft`（参数高效微调）  
- `trl`（SFT 训练）  

并验证 GPU 可用性（本项目中 GPU 计算能力为 8.6，支持 Flash Attention 加速）。

---

### 2. 数据集分析

加载 `databricks-dolly-15k` 数据集（共 15011 条样本），核心字段：

- `instruction`（指令）  
- `context`（上下文）  
- `response`（响应）  
- `category`（类别）  

覆盖分类、问答、生成等多种任务类型，为指令微调提供丰富的监督数据。

---

### 3. Alpaca-Style 数据格式化（核心重点）

Alpaca 风格通过固定模板构建 `"指令 - 输入 - 响应"` 的结构化数据，使模型学习 `"指令与响应的映射关系"`，是本次学习的核心内容。

#### 格式化逻辑

将原始数据中的 `response` 作为模型的 "输入参考"，`instruction` 作为模型需要学习生成的 "目标输出"，通过模板引导模型理解任务逻辑：

```python
def format_instruction(sample_data):
    return f"""### Instruction:
Use the Input below to create an instruction, which could have been used to generate the input using an LLM. 

### Input:
{sample_data['response']}

### Response:
{sample_data['instruction']}
"""
```

#### 示例效果

```markdown
### Instruction:
Use the Input below to create an instruction, which could have been used to generate the input using an LLM.

### Input:
In 1403, during the Ming dynasty, Beijing got its current name.

### Response:
Given this paragraph about the history of Beijing, when did Beijing get the current name?
```

#### 核心作用
##### 通过统一模板，将分散任务转化为模型可理解的 "从响应反推指令" 的标准化任务，强化模型对 "指令意图" 的理解能力，为后续遵循用户指令打下基础。

####  应用场景与优势

#####  跨任务通用性
- 不论是问答、摘要、分类还是生成任务，都可以统一为 **"指令-输入-响应"** 格式，便于模型在不同任务间迁移学习。

##### 提升生成质量
- 模型不仅学会生成答案，还能理解指令的意图，减少偏离用户需求的输出。

##### 便于数据扩展
- 可通过自动化脚本将现有文本数据批量转换为 Alpaca 风格，快速构建微调数据集。

##### 增强可解释性
- 每条样本都明确标注 **Instruction**、**Input**、**Response**，方便团队分析模型行为和调优策略。


---

### 4. 高效训练技术实践

#### 注意力机制优化对比

| 实现方式          | 优化描述                       | 显存   | 速度                  |
|-----------------|-------------------------------|--------|---------------------|
| **flash-attn**   | 高度 fused kernel，减少 memory IO | 更省显存 | **快很多**（2～4 倍，长序列时更明显） |
| **PyTorch SDPA** | 有一定优化，但没有 flash-attn 深 | 中等   | 中等                  |
| **普通 attention** | 无优化                          | 最耗显存 | 最慢                  |


#### 量化与参数高效微调

- 采用 4bit 量化（`bitsandbytes`）加载 LLaMA 2-7B 模型，大幅降低显存占用  
- 结合 QLoRA 技术，仅训练 0.12% 的模型参数（约 838 万），在有限资源下实现高效微调  

---

### 5. 训练与推理流程

- **训练配置**：
  - 使用 `SFTTrainer`  
  - 演示训练 1 轮（实际建议 3 轮）  
  - 批次大小 3，学习率 2e-4  
  - 最终训练损失稳定在 1.3 左右  

- **推理验证**：
  - 加载基础模型与 LoRA 适配器  
  - 构建 Alpaca 风格提示词生成指令  
  - 结果与真实指令匹配度较高，验证微调效果  

---

## 作业总结

### 核心收获

1. **Alpaca 风格的关键价值**：通过结构化模板将原始数据转化为标准化指令任务，强化模型理解 `"指令-响应"` 的能力  
2. **高效训练实践意义**：Flash Attention 与量化技术结合，可显著降低显存消耗并提升训练速度  
3. **端到端流程掌握**：从数据处理、模型配置到训练执行、推理验证，完整掌握 LLaMA 2 指令微调方法  


---

## 文件说明

- `llama2_instruction_tuning.ipynb`：模型微调完整流程，含数据格式化、模型配置、训练执行  
- `llama2_inference.ipynb`：微调后模型的推理验证，对比生成结果与真实指令


---

老师的示例代码中，使用了 SFTTrainer ，SFTTrainer 需要格式化的训练数据，Alpaca 风格正好符合 {"instruction","input","output"} 的结构。
所以可以说 SFTTrainer 和 Alpaca 风格是非常契合的“绝配”组合。

# SFTTrainer

**SFTTrainer** 是一种专门用于大语言模型（LLM）指令微调（Supervised Fine-Tuning, SFT）的训练工具或训练器。它的设计目的是让模型更好地理解用户指令，并生成符合预期的文本输出。相比普通的 Hugging Face `Trainer`，SFTTrainer 针对生成任务做了优化。

## 核心特点

### 1. 专注于指令微调
- 处理格式化数据：`{"instruction": "...", "input": "...", "output": "..."}`  
- 自动拼接 instruction + input + output，并生成 labels  
- 对输出部分计算损失（loss masking），避免惩罚输入内容  

### 2. 数据预处理和增强
- 自动 tokenize 输入和输出  
- 可以添加元数据或做特殊字段处理  
- 支持 batch 处理、padding 和 attention mask  

### 3. 训练优化
- 支持 gradient accumulation、低显存训练  
- 集成量化训练（bitsandbytes）、LoRA/PEFT 方案  
- 管理学习率、优化器和 checkpoint 保存  

### 4. 输出和评估
- 保存的模型可直接用于生成任务，如问答或聊天  
- 支持评估 loss、perplexity 等生成相关指标  

### 5. 适用场景
- 将基础大模型微调成聊天模型或指令理解模型  
- 小样本领域微调，提高特定任务表现  
- 在显存受限的硬件上做生成任务微调  

## 总结
SFTTrainer 的核心价值在于 **简化指令微调流程**，自动处理生成任务特有的训练需求，并支持低显存训练，使大模型微调变得更高效、方便。

---

# SFTTrainer 与 Alpaca 风格

## 1️⃣ Alpaca 风格简介

**Alpaca-Style** 是一种指令微调的数据格式/风格，由斯坦福 Alpaca 项目提出。  

数据形式通常如下：

```json
{
  "instruction": "翻译这句话为英文",
  "input": "我喜欢学习",
  "output": "I like studying"
}
```

### 核心理念：统一各种任务为 “指令-输入-输出” 的形式，让模型可以通用地理解不同指令。

### 数据集本身较小，但可以用于 SFT 微调大模型。

## 2️⃣ SFTTrainer 与 Alpaca 风格的关系


- 数据输入	SFTTrainer 需要格式化的训练数据，Alpaca 风格正好符合 {"instruction","input","output"} 的结构
- 训练目标	Alpaca 风格数据用于训练模型理解指令和生成响应，SFTTrainer 是执行训练的工具
- 搭配方式	SFTTrainer + Alpaca 风格数据 → 快速得到指令微调后的大语言模型
- 可扩展性	Alpaca 风格只是数据规范，可以用在其他 SFT 训练器（如普通 Trainer + 输出 masking）

---

```python
def format_instruction(sample_data):
    """
    Formats the given data into a structured instruction format.

    Parameters:
    sample_data (dict): A dictionary containing 'response' and 'instruction' keys.

    Returns:
    str: A formatted string containing the instruction, input, and response.
    """
    # Check if required keys exist in the sample_data
    if 'response' not in sample_data or 'instruction' not in sample_data:
        # Handle the error or return a default message
        return "Error: 'response' or 'instruction' key missing in the input data."

    return f"""### Instruction:
Use the Input below to create an instruction, which could have been used to generate the input using an LLM. 
 
### Input:
{sample_data['response']}
 
### Response:
{sample_data['instruction']}
"""

```


## 老师的代码实际上这里做了一个 **反向指令生成（Reverse Instruction Generation）** ，这是一个在大语言模型训练里越来越流行的思路。

#### 反向指令生成 (Reverse Instruction Generation)
```scss
已有输出 (Response) → 模型 → 预测指令 (Instruction)
```

### 1️⃣ 定义

**反向指令生成** 是指：

已知一个模型输出（Response 或生成文本），训练模型去预测或生成可能对应的指令（Instruction）。

换句话说，就是 **从结果反推问题**。

- **正向微调**：Instruction → Input → Response  
- **反向微调**：Response → Instruction  

---

### 2️⃣ 为什么要做

#### 扩充指令数据集
- 标注好的指令-输入-输出数据集通常有限  
- 已有的输出文本本身是资源，通过反向生成指令，可以创建更多训练样本  

#### 增强模型理解能力
- 模型不仅能完成指令，也能理解已有文本背后的意图  
- 对“推理、抽象”类任务特别有帮助  

#### 自监督训练
- 对于没有明确指令标注的大规模文本数据，可以利用已有文本生成潜在指令  
- 降低人工标注成本  
