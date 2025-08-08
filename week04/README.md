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
### 📊 **模型对比结果示例**
| 输入问题 | 基准模型（原始 ChatGLM3-6B 等常规模型）输出 | 错误数据集微调模型输出 | 过拟合模型输出 |
|----------|-------------------------------------------|------------------------|----------------|
| 解释下乾卦是什么？ | 乾卦是《周易》六十四卦之一，上乾下乾，乾象征天，蕴含刚健、元始、主宰等意义，代表积极向上、自强不息的精神内核。从卦理看，有开创、领导、尊贵等解读方向，正如 “乾卦为天，体现刚健中正之道，对应事物开端与蓬勃发展趋势，君子观乾卦象，当效法天道自强不息” 。 | 错误理解乾卦概念，输出类似 “乾卦是一种古代水利相关的卦象，用于指导农田灌溉”（因错误数据集误导，完全偏离乾卦实际含义） | 刻板复述训练数据中乾卦的固定解释，如 “乾卦，上乾下乾，天也，刚健为质，有元始、亨通、有利、正固之德，君子法之自强不息”，严格依照训练表述，无灵活拓展。 |
| 解释下地水师卦 | 师卦为《周易》卦名，坎下坤上，坎象征水、险，坤象征地、顺，整体象征军队、兵众，涉及出兵作战、指挥带兵等含义，也蕴含用兵需顺应时势、师出有名的道理，即 “师卦体现军事行动相关哲理，君子观之当懂得容民畜众、谨守正道用兵” 。 | 错误输出，如 “地水师卦是关于地质勘探的卦象，可预测地下水资源分布”（受错误数据集影响，歪曲师卦本义） | 机械重复训练数据里师卦的描述，像 “师卦，坎下坤上，象征军队，《序卦》言讼必有众起故受之以师，《象》曰地中有水师，君子以容民畜众”，严格按训练文本复述，无法灵活阐释。 |
| 解释下天水讼卦 | 讼卦属《周易》卦象，乾下坎上，乾为天、刚健，坎为水、险陷，象征争讼、争辩，代表事物发展中出现矛盾冲突待解决，同时提醒人们要防讼、止讼，比如 “讼卦体现争讼之象，君子观之当做事谋始，避免陷入无谓争讼” 。 | 错误输出，例如 “天水讼卦是天气预报相关的卦象，可预测降雨天气”（因错误数据集干扰，曲解讼卦含义） | 生硬复现训练数据中讼卦的内容，如 “讼卦，乾下坎上，天与水违行象征争讼，《序卦》云物不可以终通故受之以讼，《象》曰天与水违行讼，君子以作事谋始”，严格遵循训练文本，缺乏灵活解读。 |

---

## 💡 总结
1. **数据集质量直接决定模型效果**：错误数据集训练的模型在多数情况下无法给出合理回答。  
2. **过拟合会导致泛化能力下降**：虽然训练集上的表现极好，但对未见过的问题表现较差。  
3. **基准模型在泛化与稳定性上更均衡**。
4. **gen_dataset_deepseek.py** 是转换成python后的文件
