from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import numpy as np


class Reranker:
    def __init__(self, model, tokenizer, base_data_path):
        self.model = model
        self.tokenizer = tokenizer
        self.df_base = pd.read_csv(base_data_path)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

    def get_score(self, query, doc):
        with torch.no_grad():
            self.model.eval()
            inputs = self.tokenizer(query, doc, return_tensors='pt', padding=True, truncation=True)
            if torch.cuda.is_available():
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
            outputs = self.model(**inputs)
            return (outputs.logits.squeeze().cpu().numpy())
    def rank(self, query, top_k):
        scores = []
        docs = self.df_base['Văn bản'].tolist()
        for doc in docs:
            score = self.get_score(query, doc)
            scores.append(score)
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        rank_result = {}
        for i in range(len(docs)):
            rank_result[i+1] = scores[i]
        rank_result = dict(sorted(rank_result.items(), key=lambda x: x[1], reverse=True)[:top_k])
        return rank_result
    def full_query_search(self, data_path, top_k):
        df_data = pd.read_excel(data_path)
        results = []
        for index, row in df_data.iterrows():
            query = df_data.iloc[index]['Query']
            query = query.lower()
            ranked_indices = self.rank(query, top_k)
            results.append(ranked_indices)
        return results
if __name__ == "__main__":
    MODEL_SAVE_PATH = "Vietnamese_Reranker_finetuned"
    model= AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_PATH)
    df_test = pd.read_excel("test_data_proptit.xlsx")
    top_k = 5
    reranker = Reranker(model, tokenizer, "CLB_PROPTIT.csv")
    for index, row in df_test.iterrows():
        query = df_test.iloc[index]['Query']
        query = query.lower()
        print(f"{index}: {query} --- ranked_indices = {reranker.rank(query, top_k)}")