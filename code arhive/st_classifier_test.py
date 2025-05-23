import os
import pandas as pd
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from datasets import Dataset
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers import CrossEncoder
'''
model = SentenceTransformer("query_classifier")

text = "Как рассчитать налоговый вычет при покупке квартиры?"
test_dataset = Dataset.from_dict({
    "sentence": [text],
})

r = model(test_dataset)
print(r)

'''

# E.g. 0: sports, 1: economy, 2: politics
text = "Как рассчитать налоговый вычет при покупке квартиры?"

test_dataset = Dataset.from_dict({
    "sentence": [text],
})

# model = SentenceTransformer("query_classifier")
model = SentenceTransformer(os.path.join("training_output", "checkpoint-195"))
loss = losses.BatchSemiHardTripletLoss(model)
trainer = SentenceTransformerTrainer(
    model=model,
    loss=loss
)

print(test_dataset)
r = trainer.predict(test_dataset)
# r = trainer.predict(text)
print(r)
