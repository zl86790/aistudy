# LLM Model Quantization and Fine-tuning (第三周作业)

本项目展示了使用 **AutoGPTQ**、**AWQ** 和 **QLoRA** 三种量化/微调方法对大型语言模型的处理实践。根据硬件显存限制，选用了相对较小的模型进行实验与验证。

---

## 📌 作业一：模型量化实验

### ✅ 1. 使用 GPTQ 量化 OPT-6.7B 模型

- 使用了课程提供的代码：[AutoGPTQ_opt-2.7b.ipynb](https://github.com/DjangoPeng/LLM-quickstart/blob/main/quantization/AutoGPTQ_opt-2.7b.ipynb)
- 实际运行的文件：`01_AutoGPTQ_opt-2.7b.ipynb`
  - 由于显存限制，使用了 facebook/opt-1.3b 模型进行测试。
- 实际运行的文件：`01_AutoGPTQ_opt-6.7b.ipynb`
  - 在Quadro6000 24G显存上，使用了 facebook/opt-6.7b 模型进行测试。

### ✅ 2. 使用 AWQ 量化 OPT-2.7B 模型

- Facebook OPT 模型地址：https://huggingface.co/facebook?search_models=opt
- 实际运行的文件：`02_AWQ-opt-TinyLlama-1.1B-Chat-v1.0.ipynb`
  - 由于硬件资源有限，选择了轻量级模型 [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) 进行 AWQ 量化测试。
 
### ✅ 3. 使用 BitsAndBytes 量化 OPT-2.7B 模型 
通过查阅资料了解到 BitsAndBytes（bnb）可以在不重新训练或量化的情况下，在线加载时自动进行量化。而且 bnb 和模型的兼容性比较好，是HuggingFace官方提供的。
所以也进行了尝试。
- 实际运行的文件：`02_AWQ-opt-2.7b-bnb.ipynb`
  - 使用 **BitsAndBytes** 在加载模型时，直接把权重转化为 4bit进行量化，适合快速部署低内存推理。

### ✅ 4. 使用 LLM int8 量化 OPT-2.7B 模型 
和 AutoAWQ 相比，LLM int8 量化更简单、易用，显存节省约一半，虽然压缩率不及 AutoAWQ 激进，但能更好地保持模型精度和稳定性。
- 实际运行的文件：`02_AWQ-opt-2.7b-LLM.int8.ipynb`
  - Int8 量化原理是将模型权重从高精度浮点数压缩到 8 位整数，通过降低数值表示精度来减少显存占用和计算开销，同时尽量保留模型性能。


---

## 📌 作业二：QLoRA 微调实验

根据硬件资源情况，在 [AdvertiseGen 数据集](https://huggingface.co/datasets/AdvertiseGen) 上使用 QLoRA 微调 ChatGLM3-6B 模型，并观察 loss 变化及微调效果。

### ✅ 小样本实验

- 使用老师提供的小样本配置（非10K）
- 文件：
  - `03_peft_qlora_chatglm.ipynb`
  - `04_peft_chatglm_inference.ipynb`
 
### ✅ max_steps=200 样本实验

- 学习过程中了解到可以使用 max_steps 参数来控制 训练过程最多执行多少个梯度更新（training steps），所以也进行了尝试。
- 开启 shuffle 之后，直接使用 max_steps=1790 可能会导致训练样本重复，这是和直接设置 num_train_epochs=1 不太一样的地方。
- 文件：
  - `03_peft_qlora_chatglm_maxsteps200.ipynb`
  - `04_peft_chatglm_inference_maxsteps200.ipynb`

### ✅ 10K 样本实验

- 使用10K样本进行完整微调，观察 loss 曲线及输出效果差异
- 文件：
  - `03_peft_qlora_chatglm_10k_examples.ipynb`
  - `04_peft_chatglm_inference_10k_examples.ipynb`

---
