import os
import pandas as pd
from transformers import TextClassificationPipeline

from adapters import AutoAdapterModel

from transformers import AutoTokenizer

test_df = pd.read_csv(os.path.join("data", "labled_data_for_testing.csv"), sep="\t")
print(test_df)

test_dicts = test_df.to_dict(orient="records")
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
model = AutoAdapterModel.from_pretrained("./train_model")
model.train_adapter("queries_4classes") # Магическая активация адаптера

classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k=1)

texts = [d["query"] for d in test_dicts[:500]]
print(classifier(texts))

test_results = []
for num, d in enumerate(test_dicts):
    class_res = classifier(d["query"])
    res_d = class_res[0][0]
    print(num, "query:", d["query"], "label:", res_d["label"], "score:", res_d["score"])
    test_results.append({**d, **res_d})

test_results_df = pd.DataFrame(test_results)
print(test_results_df)

test_results_df.to_csv(os.path.join("data", "labled_data_with_test_results.csv"), sep="\t")