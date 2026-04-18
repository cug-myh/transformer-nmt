# Transformer 英中翻译项目学习笔记（从0到Tokenizer）

本文记录了从项目初始化到完成数据清洗与Tokenizer训练的完整流程，适用于基于 Transformer 的机器翻译（NMT）入门项目。

---

# 🧠 一、项目初始化

## 1. 创建项目目录

```bash
mkdir transformer-nmt
cd transformer-nmt
```

## 2. 初始化 Git 仓库

```bash
git init
```

## 3. 配置 Git 用户信息

```bash
git config --global user.name "your_name"
git config --global user.email "your_email"
```

👉 作用：用于标识提交者身份（必须配置，否则无法 commit）

---

# 📦 二、创建基础项目结构

推荐结构：

```text
transformer-nmt/
├── data/
│   ├── raw/
│   └── processed/
├── utils/
├── train.py
├── infer.py
├── config.py
└── requirements.txt
```

创建目录：

```bash
mkdir -p data/raw data/processed utils
```

---

# 📊 三、数据集准备

数据集格式（cmn.txt）：

```text
Hi.\t嗨。\tmetadata
Hi.\t你好。\tmetadata
Run.\t你用跑的。\tmetadata
```

👉 结构：

```
英文 \t 中文 \t 版权信息
```

---

# 🧼 四、数据清洗（Preprocess）

## 目标

将原始数据变为：

```text
英文 \t 中文
```

---

## 脚本：utils/preprocess.py

```python
import re

input_path = "data/raw/cmn.txt"
output_path = "data/processed/cmn_clean.txt"

def clean_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

count = 0

with open(input_path, "r", encoding="utf-8") as f, \
     open(output_path, "w", encoding="utf-8") as out:

    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue

        en = clean_text(parts[0])
        zh = clean_text(parts[1])

        if len(en) < 1 or len(zh) < 1:
            continue

        out.write(en + "\t" + zh + "\n")
        count += 1

print("Done! total pairs:", count)
```

---

## 输出结果

```text
Hi.\t嗨。
Hi.\t你好。
Run.\t你用跑的。
```

---

# 🔤 五、Tokenizer准备数据

## 为什么需要 tokenizer？

Transformer 不能处理文本，只能处理数字。

例如：

```text
I love you → [12, 45, 78]
```

---

## 构造训练数据

### utils/make_sp_data.py

```python
input_file = "data/processed/cmn_clean.txt"
output_file = "data/processed/sp_input.txt"

with open(input_file, "r", encoding="utf-8") as f, \
     open(output_file, "w", encoding="utf-8") as out:

    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue

        en, zh = parts
        out.write(en + "\n")
        out.write(zh + "\n")

print("done")
```

👉 转换结果：

```
英文
中文
英文
中文
```

---

# 🧠 六、训练 SentencePiece Tokenizer

## 安装

```bash
pip install sentencepiece
```

---

## 训练命令

```bash
spm_train \
--input=data/processed/sp_input.txt \
--model_prefix=tokenizer \
--vocab_size=8000 \
--model_type=bpe
```

---

## 参数说明

| 参数           | 含义        |
| ------------ | --------- |
| input        | 训练文本      |
| model_prefix | 输出文件名前缀   |
| vocab_size   | 词表大小      |
| model_type   | 分词算法（BPE） |

---

## 输出文件

```text
tokenizer.model   ← 模型
tokenizer.vocab   ← 词表
```

---

# 🧠 七、Tokenizer作用总结

Tokenizer 的本质：

👉 将文本 → 子词 → 数字ID

示例：

```text
I love you → [23, 456, 89]
```

---

# 🚀 八、当前项目进度

你已经完成：

✔ Git项目初始化
✔ 数据上传服务器
✔ 数据清洗（preprocess）
✔ tokenizer训练数据准备
✔ SentencePiece tokenizer训练

---

# 🔥 九、下一步（关键）

接下来进入 Transformer 核心模块：

## 👉 Dataset & DataLoader

* padding
* batch处理
* mask机制

## 👉 Transformer模型搭建

* embedding
* attention
* encoder/decoder

---

# 🎯 总结

当前阶段你已经完成：

> 从“原始文本” → “可训练tokenizer系统”

这是 NLP / Transformer 项目的第一大核心里程碑。

---

# 🧪 十、Tokenizer 测试脚本（重要）

训练完 tokenizer 后，我们需要验证它是否真的能正常工作。

## 📌 新建测试脚本

```bash
utils/test_tokenizer.py
```

---

## ✨ 测试代码

```python
import sentencepiece as spm

# 加载训练好的模型
sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")

# 测试句子
text_en = "I love you"
text_zh = "我爱你"

# 编码（text -> ids）
en_ids = sp.encode(text_en, out_type=int)
zh_ids = sp.encode(text_zh, out_type=int)

print("英文原句:", text_en)
print("英文token ids:", en_ids)
print("
中文原句:", text_zh)
print("中文token ids:", zh_ids)

# 解码（ids -> text）
print("
英文解码:", sp.decode(en_ids))
print("中文解码:", sp.decode(zh_ids))
```

---

## 🚀 运行测试

```bash
python utils/test_tokenizer.py
```

---

# 🧠 十一、你应该看到的输出效果

类似：

```text
英文原句: I love you
英文token ids: [12, 45, 89]

中文原句: 我爱你
中文token ids: [102, 88, 301]

英文解码: I love you
中文解码: 我爱你
```

---

# 🎯 十二、这个测试在验证什么？

✔ tokenizer 是否正常加载
✔ encode 是否正确（文本→数字）
✔ decode 是否正确（数字→文本）
✔ 中英文是否都支持

---

# 🚀 下一步（非常关键）

如果测试通过，下一阶段进入：

🔥 Dataset & DataLoader（Transformer输入结构）

* padding
* batch
* attention mask
