from torch.utils.data import DataLoader
from utils.dataset import TranslationDataset
from utils.collate_fn import collate_fn

dataset = TranslationDataset(
    file_path="data/processed/cmn_clean.txt",
    tokenizer_path="tokenizer.model"
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

for en, zh in loader:
    print(en)
    print(zh)

    print(en.shape, zh.shape)
    break