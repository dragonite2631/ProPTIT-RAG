from networkx import hits
from rank_bm25 import BM25Okapi, BM25Plus, BM25L
import pandas as pd
import numpy as np
class BM25:

    def __init__(self, file_clb_proptit, file_train, bm25_type):
        self.df_clb = pd.read_csv(file_clb_proptit)
        self.df_train = pd.read_excel(file_train)
        self.corpus = self.df_clb['Văn bản']
        self.tokenized_corpus = [doc.split(" ") for doc in self.corpus]
        if bm25_type == "BM25Okapi":
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        elif bm25_type == "BM25Plus":
            self.bm25 = BM25Plus(self.tokenized_corpus)
        elif bm25_type == "BM25L":
            self.bm25 = BM25L(self.tokenized_corpus)

    def rank(self, query, top_k):
        tokenized_query = query.split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        rank_result = {idx + 1: scores[idx] for idx in range(len(scores))}
        max_score = np.max(list(rank_result.values()))
        min_score = np.min(list(rank_result.values()))
        if max_score - min_score == 0: rank_result = {k: 1.0 for k,v in rank_result.items()}
        else: rank_result = {k: (v - min_score) / (max_score - min_score) for k, v in rank_result.items()}
        rank_result = dict(sorted(rank_result.items(), key=lambda x: x[1], reverse=True)[:top_k])
        return rank_result
    
    def full_query_search(self, data_path, top_k, save_path):
        df_data = pd.read_excel(data_path)
        results = []
        for index, row in df_data.iterrows():
            query = df_data.iloc[index]['Query']
            query = query.lower()
            ranked_indices = self.rank(query, top_k)
            results.append(ranked_indices)
        doc_ids = [list(results[i].keys()) for i in range(len(results))]
        scores = [list(results[i].values()) for i in range(len(results))]
        for ids in doc_ids:
            for i in range(len(ids)):
                ids[i] = str(ids[i])
        for score in scores:
            for i in range(len(score)):
                score[i] = str(score[i])
        df_save = pd.DataFrame({"doc_ids": [",".join(doc_ids[i]) for i in range(len(doc_ids))],
                                "scores":[",".join(scores[i]) for i in range(len(scores))]})
        df_save.to_excel(save_path)
        return results
if __name__ == "__main__":
    bm25_list = [ "BM25L", "BM25Plus", "BM25Okapi"]
    top_k = 60
    file_clb_proptit = "CLB_PROPTIT.csv"  
    file_train = "train_data_proptit.xlsx"
    df_train = pd.read_excel(file_train)
    bm25 = BM25(file_clb_proptit, file_train, "BM25L")
    bm25.full_query_search(file_train,top_k, "bm25_result_train.xlsx")
