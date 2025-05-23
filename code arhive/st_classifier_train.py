"""
https://sbert.net/docs/package_reference/sentence_transformer/losses.html#denoisingautoencoderloss
"""

import os
import pandas as pd
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from datasets import Dataset

data_df = pd.read_csv(os.path.join("data", "labled_data.csv"), sep="\t")
print(data_df)

sentences = data_df["query"].to_list()
labels = data_df["label"].to_list()

model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
"""
Dataset({
    features: ['sentence'],
    num_rows: 1
})
"""

# E.g. 0: sports, 1: economy, 2: politics
train_dataset = Dataset.from_dict({
    "sentence":sentences,
    "label": labels,
})

loss = losses.BatchSemiHardTripletLoss(model)
args = SentenceTransformerTrainingArguments(num_train_epochs=5, 
                                            output_dir="training_output")

trainer = SentenceTransformerTrainer(
    args=args,
    model=model,
    train_dataset=train_dataset,
    loss=loss,
)

trainer.train()

final_output_dir = "query_classifier"
trainer.save_model(final_output_dir)
model.save_pretrained(final_output_dir)

text = "Как рассчитать налоговый вычет при покупке квартиры?"
trainer.predict(text)