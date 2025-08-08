# ChatGLM3-6B QLoRA 微调与数据集对比实验

本项目基于 [DjangoPeng/LLM-quickstart](https://github.com/DjangoPeng/LLM-quickstart) 的 QLoRA 微调流程，对 **ChatGLM3-6B** 模型进行多轮不同数据集的训练与推理对比，探索数据集质量与模型效果之间的关系。

## 📂 项目结构

```
.
├── 01_qlora_chatglm3_timestamp.ipynb                 # 第一次微调
├── 01_qlora_chatglm3_timestamp-error-dataset.ipynb   # 使用错误数据集微调
├── 01_qlora_chatglm3_timestamp-lizhe-dataset-overfit.ipynb  # 过拟合训练
├── 02_chatglm_inference.ipynb                        # 模型推理与对比
├── 03_gen_dataset_deepseek.ipynb                     # 使用 Deepseek 生成新数据集
├── data/
│   ├── zhouyi_dataset_20240118_163659.csv
│   ├── zhouyi_dataset_20240118_163659_content_error.csv
│   └── zhouyi_train_dataset_lizhe.csv
└── README.md
```

---

## 🚀 实验步骤

### **第一步：基准训练**
参考微调代码：  
[qlora_chatglm3_timestamp.ipynb](https://github.com/DjangoPeng/LLM-quickstart/blob/main/chatglm/qlora_chatglm3_timestamp.ipynb)

- 数据集：`zhouyi_dataset_20240118_163659.csv`（老师提供）
- 训练脚本：`01_qlora_chatglm3_timestamp.ipynb`
- 训练轮数：3
- 输出模型：  
  ```
  chatglm3-6b-epoch3-20250808_233529
  ```

---

### **第二步：错误数据集训练**
- 数据集：  
  将原有数据集中的 **回答** 打乱生成错误版本  
  `zhouyi_dataset_20240118_163659_content_error.csv`
- 训练脚本：`01_qlora_chatglm3_timestamp-error-dataset.ipynb`
- 输出模型：  
  ```
  chatglm3-6b-epoch3-20250809_002946-error-dataset
  ```

---

### **第三步：过拟合实验**
- 数据集生成：  
  使用 `03_gen_dataset_deepseek.ipynb` 调用 Deepseek API 生成新数据集  
  `zhouyi_train_dataset_lizhe.csv`
- 训练脚本：`01_qlora_chatglm3_timestamp-lizhe-dataset-overfit.ipynb`
- 训练轮数：50（过拟合）
- 输出模型：  
  ```
  chatglm3-6b-epoch50-20250809_015451-lizhe
  ```

---

### **第四步：推理与对比**
- 推理脚本：`02_chatglm_inference.ipynb`
- 功能：加载不同版本的模型，使用相同输入进行推理，并输出对比结果



---

## 💡 总结
1. **数据集质量直接决定模型效果**：错误数据集训练的模型在多数情况下无法给出合理回答。  
2. **过拟合会导致泛化能力下降**：虽然训练集上的表现极好，但对未见过的问题表现较差。  
3. **基准模型在泛化与稳定性上更均衡**。
4. **gen_dataset_deepseek.py** 是转换成python后的文件
