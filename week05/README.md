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

#### 示例效果（显示原 Markdown 内容）

```markdown
### Instruction:
Use the Input below to create an instruction, which could have been used to generate the input using an LLM.

### Input:
1403年明朝时期，北京获得了现在的名称

### Response:
根据这段关于北京历史的文字，北京何时获得现在的名称？
```

#### 核心作用

通过统一模板，将分散任务转化为模型可理解的 `"从响应反推指令"` 的标准化任务，强化模型对 `"指令意图"` 的理解能力，为后续遵循用户指令打下基础。

---

### 4. 高效训练技术实践

#### 注意力机制优化对比

| 实现方式          | 优化描述                       | 显存   | 速度                  |
|-----------------|-------------------------------|--------|---------------------|
| **flash-attn**   | 高度 fused kernel，减少 memory IO | 更省显存 | **快很多**（2～4 倍，长序列时更明显） |
| **PyTorch SDPA** | 有一定优化，但没有 flash-attn 深 | 中等   | 中等                  |
| **普通 attention** | 无优化                          | 最耗显存 | 最慢                  |

> 本项目 GPU 支持 Flash Attention，但演示环境限制未启用，实际场景中建议优先使用以提升效率。

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

### 待优化方向

- 推理结果偶尔存在冗余信息，可通过调整生成参数（如降低温度系数）或增加后处理逻辑优化  
- 扩展训练数据规模并延长训练轮次，进一步提升模型对复杂指令的理解能力  

---

## 文件说明

- `llama2_instruction_tuning.ipynb`：模型微调完整流程，含数据格式化、模型配置、训练执行  
- `llama2_inference.ipynb`：微调后模型的推理验证，对比生成结果与真实指令  
