from rank_bm25 import BM25Okapi, BM25Plus, BM25L
import pandas as pd
import numpy as np
from reranker_search import Reranker
from bm25_search import BM25
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from llm_local import LLM
class HybridSearch:
    def __init__(self, bm25_model_type, reranker, llm_model, base_data_path, train_path, a=0.17, b=0.5):
        self.df_base = pd.read_csv(base_data_path)
        self.reranker = reranker
        self.bm25_model = BM25(base_data_path, train_path, bm25_model_type)
        self.llm_model = llm_model
        self.df_data = pd.read_excel(train_path)
        self.a = a
        self.b = b

    def search(self, query, top_k_phase1 = 60, top_k_phase2 = 10, top_k_final = 5):

        query = query.lower()
        phase1_result = self.bm25_model.rank(query, top_k_phase1)
        reranker_scores = {}
        reranker_result = {}
        phase2_result = {}
        llm_scores = {}
        llm_result = {}
        final_result = {}

        for idx in phase1_result:
            article = self.df_base['Văn bản'].iloc[idx - 1].lower()
            reranker_score = self.reranker.get_score(query, article)
            reranker_scores[idx] = reranker_score
        min_score = np.min(list(reranker_scores.values()))
        max_score = np.max(list(reranker_scores.values()))
        if max_score - min_score == 0: reranker_result = {k: 1.0 for k in reranker_scores.keys()}
        reranker_result = {k: (v - min_score) / (max_score - min_score) for k, v in reranker_scores.items()}
        for idx in phase1_result:
            phase2_result[idx] = self.a * phase1_result[idx] + (1-self.a) * reranker_result[idx]
        phase2_result = dict(sorted(phase2_result.items(), key=lambda x: x[1], reverse=True)[:top_k_phase2])
        
        for idx in phase2_result:
            article = self.df_base['Văn bản'].iloc[idx - 1]
            llm_score = self.llm_model.get_score(query, article)
            llm_scores[idx] = llm_score
        min_score = np.min(list(llm_scores.values()))
        max_score = np.max(list(llm_scores.values()))
        if max_score - min_score == 0: llm_result = {k: 1.0 for k in llm_scores.keys()}
        else: llm_result = {k: (v - min_score) / (max_score - min_score) for k, v in llm_scores.items()}
        for idx in phase2_result:
            final_result[idx] = self.b * phase2_result[idx] + (1-self.b) * llm_result[idx]
        final_result = dict(sorted(final_result.items(), key=lambda x: x[1], reverse=True)[:top_k_final])
        return final_result

    
    def full_query_search(self, save_result_file):
        df_save = pd.read_csv(save_result_file)
        retrieval_results = []
        response_results = []
        cur_length = len(df_save)
        for _, row in self.df_data.iterrows():
            if(_ < cur_length): continue
            query = row['Query']
            ranked_indices = self.search(query)
            answer_prompt = self.llm_model.generate_answer_prompt(query, list(ranked_indices.keys()))
            llm_response = self.llm_model.generate_response(answer_prompt)
            retrieval_results.append(ranked_indices)
            response_results.append(llm_response)
            doc_ids = list(ranked_indices.keys())
            scores = [item[1] for item in ranked_indices.items()]
            new_row = pd.DataFrame([{"query": query, "doc_ids": ",".join(str(idx) for idx in doc_ids),"scores" : ",".join(str(score) for score in scores),"response": llm_response}])
            df_save = pd.concat([df_save, new_row], ignore_index=True)
            df_save.to_csv(save_result_file, index=False)
            print(f"{query} / {ranked_indices.keys()} / {llm_response}")
            print("-" * 50)
        return retrieval_results, response_results


if __name__ == "__main__":
    base_data_path = "CLB_PROPTIT.csv"
    search_path = "test_data_proptit.csv"
    save_path = "test_result.csv"
    bm25_type = "BM25L"  # or "BM25Plus", "BM25L"
    reranker_path = "Vietnamese_Reranker_finetuned"
    model_name = "Qwen/Qwen3-0.6B"

    reranker = Reranker(AutoModelForSequenceClassification.from_pretrained(reranker_path),
                        AutoTokenizer.from_pretrained(reranker_path),
                        base_data_path)
    model = LLM(model_name, base_data_path)
    hybrid_search = HybridSearch(bm25_type, reranker, model, base_data_path, search_path)
    hybrid_search.full_query_search(save_path)
