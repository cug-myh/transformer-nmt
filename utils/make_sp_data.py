input_file = "data/processed/cmn_clean.txt"
output_file = "data/processed/sp_input.txt"

with open(input_file, "r", encoding="utf-8") as f, \
     open(output_file, "w", encoding="utf-8") as out:

    for line in f:
        parts = line.strip().split("\t")
        if len(parts) != 2:
            continue

        en = parts[0]
        zh = parts[1]

        # SentencePiece训练要求：一行一句
        out.write(en + "\n")
        out.write(zh + "\n")

print("完成！")