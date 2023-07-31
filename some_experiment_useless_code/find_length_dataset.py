
from datasets import load_dataset, DatasetDict, load_from_disk
import torch
from determine_batch_size import get_batch_size

preprocessed_splits = load_from_disk("./processed_datadir/wikitext-103-story-chartoken-bert-2048")
print("train length :", len(preprocessed_splits["train"]))

preprocessed_splits = load_from_disk("./processed_datadir/wikitext-103-story-chartoken-bert-8196")
print("train length :", len(preprocessed_splits["train"]))  # 这2个长度甚至是一样的