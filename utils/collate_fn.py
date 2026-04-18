import torch

# 【正确】和新tokenizer完全匹配
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

def collate_fn(batch):
    en_batch, zh_batch = zip(*batch)

    def add_bos_eos(seq):
        return torch.cat([
            torch.tensor([BOS_ID]),
            seq,
            torch.tensor([EOS_ID])
        ])

    en_batch = [add_bos_eos(s) for s in en_batch]
    zh_batch = [add_bos_eos(s) for s in zh_batch]

    en_max = max(len(s) for s in en_batch)
    zh_max = max(len(s) for s in zh_batch)

    def pad(seq, max_len):
        pad_len = max_len - len(seq)
        padding = torch.full((pad_len,), PAD_ID, dtype=torch.long)
        return torch.cat([seq, padding])

    en_batch = torch.stack([pad(s, en_max) for s in en_batch])
    zh_batch = torch.stack([pad(s, zh_max) for s in zh_batch])

    return en_batch, zh_batch