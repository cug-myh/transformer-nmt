import torch
from torch.utils.data import Dataset
import sentencepiece as spm

class TranslationDataset(Dataset):
    def __init__(self, file_path, tokenizer_path, max_len=50):
        self.data = []
        self.max_len = max_len

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue

                self.data.append((parts[0], parts[1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        en, zh = self.data[idx]

        en_ids = self.sp.encode(en, out_type=int)
        zh_ids = self.sp.encode(zh, out_type=int)

        return torch.tensor(en_ids), torch.tensor(zh_ids)