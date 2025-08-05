# LLM Model Quantization and Fine-tuning (第三周作业)

本项目展示了使用 **AutoGPTQ**、**AWQ** 和 **QLoRA** 三种量化/微调方法对大型语言模型的处理实践。根据硬件显存限制，选用了相对较小的模型进行实验与验证。

---

## 📌 作业一：模型量化实验

### ✅ 1. 使用 GPTQ 量化 OPT-6.7B 模型

- 使用了课程提供的代码：[AutoGPTQ_opt-2.7b.ipynb](https://github.com/DjangoPeng/LLM-quickstart/blob/main/quantization/AutoGPTQ_opt-2.7b.ipynb)
- 实际运行的文件：`01_AutoGPTQ_opt-2.7b.ipynb`
  由于显存限制，使用了 facebook/opt-1.3b 模型进行测试。
- 实际运行的文件：`01_AutoGPTQ_opt-6.7b.ipynb`
  在Quadro6000 24G显存上，使用了 facebook/opt-6.7b 模型进行测试。

### ✅ 2. 使用 AWQ 量化 OPT-6.7B 模型

- Facebook OPT 模型地址：https://huggingface.co/facebook?search_models=opt
- 实际运行的文件：`02_AWQ-opt-TinyLlama-1.1B-Chat-v1.0.ipynb`
- 由于硬件资源有限，选择了轻量级模型 [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) 进行 AWQ 量化测试。

---

## 📌 作业二：QLoRA 微调实验

根据硬件资源情况，在 [AdvertiseGen 数据集](https://huggingface.co/datasets/AdvertiseGen) 上使用 QLoRA 微调 ChatGLM3-6B 模型，并观察 loss 变化及微调效果。

### ✅ 小样本实验

- 使用老师提供的小样本配置（非10K）
- 文件：
  - `03_peft_qlora_chatglm.ipynb`
  - `04_peft_chatglm_inference.ipynb`

### ✅ 10K 样本实验

- 使用10K样本进行完整微调，观察 loss 曲线及输出效果差异
- 文件：
  - `03_peft_qlora_chatglm_10k_examples.ipynb`
  - `04_peft_chatglm_inference_10k_examples.ipynb`

---
