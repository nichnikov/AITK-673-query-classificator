from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from datasets import Dataset

model = SentenceTransformer("microsoft/mpnet-base")
# E.g. 0: sports, 1: economy, 2: politics
train_dataset = Dataset.from_dict({
    "sentence": [
        "He played a great game.",
        "The stock is up 20%",
        "They won 2-1.",
        "The last goal was amazing.",
        "They all voted against the bill.",
    ],
    "label": [0, 1, 0, 0, 2],
})
loss = losses.BatchSemiHardTripletLoss(model)

trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    loss=loss,
)

test_dataset = Dataset.from_dict({
    "sentence": ["He played a great game."],
    "label": [0]})

trainer.train()

r = trainer.predict(test_dataset)

print(r)