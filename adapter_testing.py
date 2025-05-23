from transformers import TextClassificationPipeline
from transformers import XLMRobertaConfig
from adapters import AutoAdapterModel
from transformers import AutoTokenizer
import numpy as np
from transformers import AutoModel, TrainingArguments, EvalPrediction
from adapters import AdapterTrainer


tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
model = AutoAdapterModel.from_pretrained("./train_model")
model.train_adapter("queries_4classes") # Магическая активация адаптера

classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

class_res = classifier("Оформление и расчет отпускных в 2025 году")
print(class_res)