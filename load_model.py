from transformers import XLMRobertaConfig
from adapters import AutoAdapterModel

config = XLMRobertaConfig.from_pretrained(
    "BAAI/bge-m3",
    num_labels=4,
)
model = AutoAdapterModel.from_pretrained(
    "BAAI/bge-m3",
    config=config,
)