import sentencepiece as spm

# ====================== 1. 训练分词模型 ======================
# 准备你的语料 txt
# model_type: unigram = glm系  | bpe = GPT系
spm.SentencePieceTrainer.train(
    input="data/processed/sp_input.txt",  # 你的语料文件
    model_prefix="tokenizer",             # 输出模型前缀
    vocab_size=8000,                      # 词表大小
    model_type="bpe",
    character_coverage=1.0,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece='<pad>',
    unk_piece='<unk>',
    bos_piece='<s>',
    eos_piece='</s>'
)

# ====================== 2. 加载模型 ======================
sp = spm.SentencePieceProcessor()
sp.load('tokenizer.model')

# ====================== 3. 文本 -> token id ======================
text_en = "I love you"
text_zh = "我爱你"

# 编码（text -> ids）
en_ids = sp.encode(text_en, out_type=int)
zh_ids = sp.encode(text_zh, out_type=int)

print("英文原句:", text_en)
print("英文token ids:", en_ids)
print("中文原句:", text_zh)
print("中文token ids:", zh_ids)

# 解码（ids -> text）
print("英文解码:", sp.decode(en_ids))
print("中文解码:", sp.decode(zh_ids))