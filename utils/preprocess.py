import re

input_path = "data/raw/cmn.txt"
output_path = "data/processed/cmn_clean.txt"

def clean_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)   # 多空格变1个
    return text

count = 0

with open(input_path, "r", encoding="utf-8") as f, \
     open(output_path, "w", encoding="utf-8") as out:

    for line in f:
        parts = line.strip().split("\t")

        # ❗必须至少两列
        if len(parts) < 2:
            continue

        en = clean_text(parts[0])
        zh = clean_text(parts[1])

        out.write(en + "\t" + zh + "\n")
        count += 1

print("完成数据清理，共有的中英文句子对数为:", count) # 完成数据清理，共有的中英文句子对数为: 29909