from reranker_search import Reranker
from llm_api import LLM
from transformers import AutoModelForSequenceClassification,  AutoTokenizer
import pandas as pd
llm_name = "openai/gpt-oss-20b"
base_data_path = "CLB_PROPTIT.csv"
reranker_path ="Vietnamese_Reranker_finetuned"
llm_model = LLM(llm_name, base_data_path)
retrieval_result = None
df_entities = None
reranker_model =Reranker(AutoModelForSequenceClassification.from_pretrained(reranker_path),
                         AutoTokenizer.from_pretrained(reranker_path),
                         base_data_path
                         )

def hit_k(file_clb_proptit, file_train_data_proptit, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train_data_proptit)

    hits = 0
    total_queries = len(df_train)

    for index, row in df_train.iterrows():
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        retrieved_docs = [int(id) for id in retrieval_result.iloc[index]["doc_ids"].split(",")]
        retrieved_docs = retrieved_docs[:k]
        ground_truth_docs =  []
        if type(ground_truth_doc) is str:
            for doc in ground_truth_doc.split(","):
                ground_truth_docs.append(int(doc))
        else:
            ground_truth_docs.append(int(ground_truth_doc))
        if any(doc in retrieved_docs for doc in ground_truth_docs):
            hits += 1
    
    return hits / total_queries if total_queries > 0 else 0


# Hàm recall@k
def recall_k(file_clb_proptit, file_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    
    ans = 0

    for index, row in df_train.iterrows():
        hits = 0
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        retrieved_docs = [int(id) for id in retrieval_result.iloc[index]["doc_ids"].split(",")]
        retrieved_docs = retrieved_docs[:k]
        ground_truth_docs =  []
        if type(ground_truth_doc) is str:
            for doc in ground_truth_doc.split(","):
                ground_truth_docs.append(int(doc))
        else:
            ground_truth_docs.append(int(ground_truth_doc))
        if any(doc in retrieved_docs for doc in ground_truth_docs):
            hits = len([doc for doc in ground_truth_docs if doc in retrieved_docs])
        ans += hits / len(ground_truth_docs) 
    return ans / len(df_train)


# Hàm precision@k
def precision_k(file_clb_proptit, file_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    ans = 0

    for index, row in df_train.iterrows():
        hits = 0
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        retrieved_docs = [int(id) for id in retrieval_result.iloc[index]["doc_ids"].split(",")]
        retrieved_docs = retrieved_docs[:k]
        ground_truth_docs =  []
        if type(ground_truth_doc) is str:
            for doc in ground_truth_doc.split(","):
                ground_truth_docs.append(int(doc))
        else:
            ground_truth_docs.append(int(ground_truth_doc))
        if any(doc in retrieved_docs for doc in ground_truth_docs):
            hits = len([doc for doc in retrieved_docs if doc in ground_truth_docs])
        ans += hits / k 
        # print("Hits / k for this query:", hits / k)
    return ans / len(df_train)


# Hàm f1@k
def f1_k(file_clb_proptit, file_train, embedding, vector_db, k=5):
    precision = precision_k(file_clb_proptit, file_train, embedding, vector_db, k)
    recall = recall_k(file_clb_proptit, file_train, embedding, vector_db, k)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

# Hàm MAP@k

def map_k(file_clb_proptit, file_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_map = 0

    for index, row in df_train.iterrows():
        hits = 0
        ap = 0
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        
        retrieved_docs = [int(id) for id in retrieval_result.iloc[index]["doc_ids"].split(",")]
        retrieved_docs = retrieved_docs[:k]
        ground_truth_docs =  []
        if type(ground_truth_doc) is str:
            for doc in ground_truth_doc.split(","):
                ground_truth_docs.append(int(doc))
        else:
            ground_truth_docs.append(int(ground_truth_doc))
        # print("Ground truth documents:", ground_truth_docs)
        
        # Tính MAP cho 1 truy vấn
        for i, doc in enumerate(retrieved_docs):
            if doc in ground_truth_docs:
                hits += 1
                ap += hits / (i + 1)
        if hits > 0:
            ap /= hits
        total_map += ap 
    return total_map / len(df_train)

# Hàm MRR@k
def mrr_k(file_clb_proptit, file_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_mrr = 0

    for index, row in df_train.iterrows():
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        retrieved_docs = [int(id) for id in retrieval_result.iloc[index]["doc_ids"].split(",")]
        retrieved_docs = retrieved_docs[:k]
        ground_truth_docs =  []
        if type(ground_truth_doc) is str:
            for doc in ground_truth_doc.split(","):
                ground_truth_docs.append(int(doc))
        else:
            ground_truth_docs.append(int(ground_truth_doc))
        for i, doc in enumerate(retrieved_docs):
            if doc in ground_truth_docs:
                total_mrr += 1 / (i + 1)
                break
    return total_mrr / len(df_train) if len(df_train) > 0 else 0

# Hàm NDCG@k
import numpy as np
def dcg_at_k(relevances, k):
    relevances = np.array(relevances)[:k]
    return np.sum((2**relevances - 1) / np.log2(np.arange(2, len(relevances) + 2)))

def ndcg_at_k(relevances, k):
    dcg_max = dcg_at_k(sorted(relevances, reverse=True), k)
    if dcg_max == 0:
        return 0.0
    return dcg_at_k(relevances, k) / dcg_max

def similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cos_sim = dot_product / (norm1 * norm2)
    return (cos_sim + 1) / 2


def ndcg_k(file_clb_proptit, file_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_ndcg = 0

    for index, row in df_train.iterrows():
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        retrieved_docs = [int(id) for id in retrieval_result.iloc[index]["doc_ids"].split(",")]
        retrieved_docs = retrieved_docs[:k]
        doc_scores = [float(score) for score in retrieval_result.iloc[index]["scores"].split(",")]
        mp_doc_score = {}
        for i in range(len(retrieved_docs)): mp_doc_score[retrieved_docs[i]] = doc_scores[i]

        ground_truth_docs = []
        if type(ground_truth_doc) is str:
            for doc in ground_truth_doc.split(","):
                ground_truth_docs.append(int(doc))
        else:
            ground_truth_docs.append(int(ground_truth_doc))
        relevances = []
        for doc in retrieved_docs:
            if doc in ground_truth_docs:
        
                similarity_score =  mp_doc_score[doc]
                if similarity_score > 0.9:
                    relevances.append(3)
                elif similarity_score > 0.7:
                    relevances.append(2)
                elif similarity_score > 0.5:
                    relevances.append(1)
                else:
                    relevances.append(0)
            else:
                relevances.append(0)
        ndcg = ndcg_at_k(relevances, k)
        # print(f"NDCG for this query: {ndcg}")
        total_ndcg += ndcg

    return total_ndcg / len(df_train) if len(df_train) > 0 else 0

# Hàm Context Precision@k (LLM Judged)

def context_precision_k(file_clb_proptit, file_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_precision = 0

    for index, row in df_train.iterrows():

        hits = 0
        query = row['Query']
        ground_truth_doc = row['Ground truth document']
        results = [int(id) for id in retrieval_result.iloc[index]["doc_ids"].split(",")]
        results = results[:k]
 
        # # TODO: viết câu query của người dùng (bao gồm document retrieval và câu query)
        context = "Content từ các tài liệu liên quan:\n"
        context += "\n".join([df_clb.iloc[result-1]["Văn bản"] for result in results])

        reply = retrieval_result.iloc[index][f"response_{k}"]
        for result in results:
            # NOTE: Các em có thể chỉnh messages_judged nếu muốn
            messages_judged =  """**Crucially, from now only use Vietnamese to response** Bạn là một trợ lý AI chuyên đánh giá độ chính xác 
            của các câu trả lời dựa trên ngữ cảnh được cung cấp. Bạn sẽ được cung cấp một ngữ cảnh, một câu hỏi và một câu trả lời từ một mô hình AI. 
            Nhiệm vụ của Bạn là đánh giá câu trả lời dựa trên ngữ cảnh và câu hỏi. 
            Nếu ngữ cảnh và câu hỏi cung cấp đủ thông tin hoặc chỉ cần một phần thông tin để trả lời câu hỏi, hãy đánh giá câu trả lời là 1. 
            Nếu không, hãy đánh giá là 0. Chỉ cần ngữ cảnh có một phần thông tin để trả lời cho một phần của câu hỏi thì cũng đánh giá là 1. 
            Nếu ngữ cảnh không liên quan gì đến câu hỏi, hãy đánh giá là 0. """
            
            # TODO: "content" sẽ lưu ngữ cảnh, câu hỏi, câu trả lời
            reply = retrieval_result.iloc[index][f"response_{k}"]
            context = df_clb.iloc[result-1]["Văn bản"] 
            messages_judged +=  f"\n **Bây giờ hãy đánh giá câu trả lời sau dựa trên ngữ cảnh và câu hỏi- LƯU Ý: Chỉ trả lời 1 hoặc 0, không giải thích gì thêm.** Ngữ cảnh: {context}\nCâu hỏi: {query} \nCâu trả lời: {reply}\nNgữ cảnh trên có liên quan đến câu hỏi không?"
            # Gọi API đến LLM Judged
            judged_response = llm_model.generate_response(messages_judged)
            print(index, judged_response)
            judged_reply = "1" if "1" in judged_response else "0"
            if judged_reply == "1":
                hits += 1
        precision = hits / k if k > 0 else 0
        total_precision += precision
    df = pd.DataFrame({"context_precision_k": [total_precision / len(df_train) if len(df_train) > 0 else 0]})
    df.to_csv(f"context_precision_k{k}.csv")
    return total_precision / len(df_train) if len(df_train) > 0 else 0


# Hàm Context Recall@k (LLM Judged)
def context_recall_k(file_clb_proptit, file_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_recall = 0

    for index, row in df_train.iterrows():
        hits = 0
        query = row['Query']
        ground_truth_doc = row['Ground truth document']

        results = [int(id) for id in retrieval_result.iloc[index]["doc_ids"].split(",")]
        results = results[:k]
        reply = row["Ground truth answer"]
        

        # NOTE: Các em có thể thay đổi messages_judged nếu muốn 
        for result in results:
            messages_judged =  """**Crucially, from now only use Vietnamese to response** Bạn là một trợ lý AI chuyên đánh giá độ chính xác 
            của các câu trả lời dựa trên ngữ cảnh được cung cấp. Bạn sẽ được cung cấp một ngữ cảnh, một câu hỏi và một câu trả lời từ một mô hình AI. 
            Nhiệm vụ của Bạn là đánh giá câu trả lời dựa trên ngữ cảnh và câu hỏi. 
            Nếu ngữ cảnh và câu hỏi cung cấp đủ thông tin hoặc chỉ cần một phần thông tin để trả lời câu hỏi, hãy đánh giá câu trả lời là 1. 
            Nếu không, hãy đánh giá là 0. Chỉ cần ngữ cảnh có một phần thông tin để trả lời cho một phần của câu hỏi thì cũng đánh giá là 1. 
            Nếu ngữ cảnh không liên quan gì đến câu hỏi, hãy đánh giá là 0. """
            
            # TODO: "content" sẽ lưu ngữ cảnh, câu hỏi, câu trả lời
            reply = retrieval_result.iloc[index][f"response_{k}"]
            context = df_clb.iloc[result - 1]["Văn bản"] 
            messages_judged +=  f"\n**Bây giờ hãy đánh giá câu trả lời sau dựa trên ngữ cảnh và câu hỏi - LƯU Ý: Chỉ trả lời 1 hoặc 0, không giải thích gì thêm.** \n Ngữ cảnh: {context}\nCâu hỏi: {query}\nCâu trả lời: {reply}\nNgữ cảnh trên có liên quan đến câu hỏi không?"
            

            judged_response = llm_model.generate_response(messages_judged)
            print(index,judged_response)
            judged_reply = "1" if "1" in judged_response else "0"
            if judged_reply == "1":
                hits += 1
        recall = hits / k if k > 0 else 0
        total_recall += recall
    df = pd.DataFrame({"context_recall_k": [total_recall / len(df_train) if len(df_train) > 0 else 0]})
    df.to_csv(f"context_recall_k{k}.csv")
    return total_recall / len(df_train) if len(df_train) > 0 else 0

def context_entities_recall_k(file_clb_proptit, file_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_recall = 0
    for index, row in df_train.iterrows():
        hits = 0
        query = row['Query']
        ground_truth_doc = row['Ground truth document']

        results = [int(id) for id in retrieval_result.iloc[index]["doc_ids"].split(",")]
        results = results[:k]
        reply = row['Ground truth answer']
        
        # messages_entities = """**Crucially, from now only use Vietnamese to response** Bạn là một trợ lý AI chuyên trích xuất các thực thể từ câu trả lời. Bạn sẽ được cung cấp một câu trả lời và nhiệm vụ của Bạn là trích xuất các thực thể từ câu trả lời đó. Các thực thể có thể là tên người, địa điểm, tổ chức, sự kiện, v.v. LƯU Ý: Hãy trả lời dưới dạng một danh sách các thực thể, không giải thích hay chèn thêm bất cứ văn bản nào.
        #         Ví dụ:
        #         Câu trả lời: Nếu bạn thuộc ngành khác bạn vẫn có thể tham gia CLB chúng mình. Nếu định hướng của bạn hoàn toàn là theo CNTT thì CLB chắc chắn là nơi phù hợp nhất để các bạn phát triển. Trở ngại lớn nhất sẽ là do bạn theo một hướng khác nữa nên sẽ phải tập trung vào cả 2 mảng nên sẽ cần cố gắng nhiều hơn.
        #         Các thực thể: ["ngành khác", "CLB", "CNTT", "mảng]
        #         Câu trả lời: Câu lạc bộ Lập Trình PTIT (Programming PTIT), tên viết tắt là PROPTIT được thành lập ngày 9/10/2011. Với phương châm hoạt động "Chia sẻ để cùng nhau phát triển", câu lạc bộ là nơi giao lưu, đào tạo các môn lập trình và các môn học trong trường, tạo điều kiện để sinh viên trong Học viện có môi trường học tập năng động sáng tạo. Slogan: Lập Trình PTIT - Lập trình từ trái tim.
        #         Các thực thể: ["Câu lạc bộ Lập Trình PTIT (Programming PTIT)", "PROPTIT", "9/10/2011", "Chia sẻ để cùng nhau phát triển", "sinh viên", "Học viện", "Lập Trình PTIT - Lập trình từ trái tim"]"""
        # # NOTE: Các em có thể thay đổi content nếu muốn
        # messages_entities += f"**Tương tự** hãy đưa ra các thực thể từ câu trả lời sau**LƯU Ý: Hãy trả lời dưới dạng một danh sách các thực thể, không giải thích hay chèn thêm bất cứ văn bản nào**\nCâu trả lời: {reply} \n"
        # # Gọi  API để trích xuất các thực thể
        # while True:
        #     try:
        #         entities  = llm_model.generate_response(messages_entities).strip().split("\n")
        #         entities = entities[0] # "["ngành khác", "CLB", "CNTT", "mảng]" -> ["ngành khác", "CLB", "CNTT", "mảng"]
        #         entities = eval(entities) if entities else []  # Chuyển đổi chuỗi thành danh sách
        #         break
        #     except Exception as e:
        #         print(e)
        #         pass
        entities = df_entities.iloc[index]["entities"]
        entities = eval(entities) if entities else []  # Chuyển đổi chuỗi thành danh sách
        print(index, entities)
        tmp = len(entities)
        for result in results:
            context = df_clb.iloc[result-1]["Văn bản"]
            for entity in entities:
                if entity.strip() in context:
                    hits += 1
                    entities.remove(entity.strip())
        total_recall += hits / tmp if tmp > 0 else 0
    df = pd.DataFrame({"context_entities_recall_k" : [total_recall / len(df_train) if len(df_train) > 0 else 0]})
    df.to_csv(f"context_entities_recall_k{k}.csv")
    return total_recall / len(df_train) if len(df_train) > 0 else 0



# Hàm tính toán tất cả metrics liên quan đến Retrieval

def calculate_metrics_retrieval(file_clb_proptit, file_train , embedding, vector_db, train):
    global df_entities
    global retrieval_result
    if train:
        df_entities = pd.read_csv("train_entities.csv")
        retrieval_result = pd.read_excel("train_result.xlsx")
    else:
        df_entities = pd.read_csv("test_entities.csv")
        retrieval_result = pd.read_excel("test_result.xlsx")
    k_values = [3,5,7]
    metrics = {
        "K": [],
        "hit@k": [],
        "recall@k": [],
        "precision@k": [],
        "f1@k": [],
        "map@k": [],
        "mrr@k": [],
        "ndcg@k": [],
        "context_precision@k": [],
        "context_recall@k": [],
        "context_entities_recall@k": []
    }
    for k in k_values:
        metrics["K"].append(k)
        metrics["hit@k"].append(round(hit_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["recall@k"].append(round(recall_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["precision@k"].append(round(precision_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["f1@k"].append(round(f1_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["map@k"].append(round(map_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["mrr@k"].append(round(mrr_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["ndcg@k"].append(round(ndcg_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["context_precision@k"].append(round(context_precision_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["context_recall@k"].append(round(context_recall_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["context_entities_recall@k"].append(round(context_entities_recall_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics_df = pd.DataFrame(metrics)
    # Lưu DataFrame vào file csv
        # if train:
        #     metrics_df.to_csv("metrics_retrieval_train.csv", index=False)
        # else:
        #     metrics_df.to_csv("metrics_retrieval_test.csv", index=False)
        # print("done---")
    # Chuyển đổi metrics thành DataFrame
    metrics_df = pd.DataFrame(metrics)
    # Lưu DataFrame vào file csv
    if train:
        metrics_df.to_csv("metrics_retrieval_train.csv", index=False)
    else:
        metrics_df.to_csv("metrics_retrieval_test.csv", index=False)
    return metrics_df



def string_presence_k(file_clb_proptit, file_train, embedding, vector_db,  k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_presence = 0

    for index, row in df_train.iterrows():
        hits = 0
        query = row['Query']
        results = [int(id) for id in retrieval_result.iloc[index]["doc_ids"].split(",")]
        results = results[:k]
        reply = row["Ground truth answer"]

        response = retrieval_result.iloc[index][f"response_{k}"]
        # messages_entities = """**Crucially, from now only use Vietnamese to response** Bạn là một trợ lý AI chuyên trích xuất các thực thể từ câu trả lời. Bạn sẽ được cung cấp một câu trả lời và nhiệm vụ của Bạn là trích xuất các thực thể từ câu trả lời đó. Các thực thể có thể là tên người, địa điểm, tổ chức, sự kiện, v.v. LƯU Ý: Hãy trả lời dưới dạng một danh sách các thực thể, không giải thích hay chèn thêm bất cứ văn bản nào.
        #         Ví dụ:
        #         Câu trả lời: Nếu bạn thuộc ngành khác bạn vẫn có thể tham gia CLB chúng mình. Nếu định hướng của bạn hoàn toàn là theo CNTT thì CLB chắc chắn là nơi phù hợp nhất để các bạn phát triển. Trở ngại lớn nhất sẽ là do bạn theo một hướng khác nữa nên sẽ phải tập trung vào cả 2 mảng nên sẽ cần cố gắng nhiều hơn.
        #         Các thực thể: ["ngành khác", "CLB", "CNTT", "mảng]
        #         Câu trả lời: Câu lạc bộ Lập Trình PTIT (Programming PTIT), tên viết tắt là PROPTIT được thành lập ngày 9/10/2011. Với phương châm hoạt động "Chia sẻ để cùng nhau phát triển", câu lạc bộ là nơi giao lưu, đào tạo các môn lập trình và các môn học trong trường, tạo điều kiện để sinh viên trong Học viện có môi trường học tập năng động sáng tạo. Slogan: Lập Trình PTIT - Lập trình từ trái tim.
        #         Các thực thể: ["Câu lạc bộ Lập Trình PTIT (Programming PTIT)", "PROPTIT", "9/10/2011", "Chia sẻ để cùng nhau phát triển", "sinh viên", "Học viện", "Lập Trình PTIT - Lập trình từ trái tim"]"""
        # # NOTE: Các em có thể thay đổi content nếu muốn
        # messages_entities += f"**Tương tự** hãy đưa ra các thực thể từ câu trả lời sau**LƯU Ý: Hãy trả lời dưới dạng một danh sách các thực thể, không giải thích hay chèn thêm bất cứ văn bản nào**\nCâu trả lời: {reply} \n"
        # # Gọi  API để trích xuất các thực thể
        # while True:
        #     try:
        #         entities  = llm_model.generate_response(messages_entities).strip().split("\n")
        #         entities = entities[0] # "["ngành khác", "CLB", "CNTT", "mảng]" -> ["ngành khác", "CLB", "CNTT", "mảng"]
        #         entities = eval(entities) if entities else []  # Chuyển đổi chuỗi thành danh sách
        #         break
        #     except Exception as e:
        #         print(e)
        #         pass
        
        entities = df_entities.iloc[index]["entities"]
        entities = eval(entities) if entities else []
        print(entities)
        for entity in entities:
            if entity.strip() in response:
                hits += 1
                # print(f"Entity '{entity.strip()}' found in response.")
        if len(entities) > 0: hits/= len(entities)
        else: hits = 0
        total_presence += hits
    df = pd.DataFrame({"string_k": [total_presence / len(df_train) if len(df_train) > 0 else 0]})
    df.to_csv("string_k.csv")
    return total_presence / len(df_train) if len(df_train) > 0 else 0


 

# Hàm Rouge-L

from rouge import Rouge
def rouge_l_k(file_clb_proptit, file_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    rouge = Rouge()
    total_rouge_l = 0

    for index, row in df_train.iterrows():
        query = row['Query']

        results = [int(id) for id in retrieval_result.iloc[index]["doc_ids"].split(",")]
        results = results[:k]
        reply = row['Ground truth answer']
 
        response = retrieval_result.iloc[index][f"response_{k}"]
        scores = rouge.get_scores(response, reply)
        rouge_l = scores[0]['rouge-l']['f']
        total_rouge_l += rouge_l
    return total_rouge_l / len(df_train) if len(df_train) > 0 else 0

# Hàm BLEU-4
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
def bleu_4_k(file_clb_proptit, file_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_bleu_4 = 0
    smoothing_function = SmoothingFunction().method1

    for index, row in df_train.iterrows():
        query = row['Query']
        results = [int(id) for id in retrieval_result.iloc[index]["doc_ids"].split(",")]
        results = results[:k]
        reply = row['Ground truth answer']
    
        response = retrieval_result.iloc[index][f"response_{k}"]
        reference = reply.split()
        candidate = response.split()
        bleu_4 = sentence_bleu([reference], candidate, smoothing_function=smoothing_function)
        total_bleu_4 += bleu_4
    return total_bleu_4 / len(df_train) if len(df_train) > 0 else 0

# Hàm Groundedness (LLM Answer - Hallucination Detection)\

def groundedness_k(file_clb_proptit, file_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_groundedness = 0

    for index, row in df_train.iterrows():
        hits = 0
        cnt = 0
        query = row['Query']
        results = [int(id) for id in retrieval_result.iloc[index]["doc_ids"].split(",")][:k]
        reply = row['Ground truth answer']
        context = "Content từ các tài liệu liên quan:\n"
        context += "\n".join([df_clb.iloc[result-1]["Văn bản"] for result in results])
        response = retrieval_result.iloc[index][f"response_{k}"]
        sentences = response.split('. ')
        for sentence in sentences:
            messages_groundedness = """**Crucially, from now only use Vietnamese to response** Bạn là một chuyên gia đánh giá Groundedness trong hệ thống RAG của một CLB lập trình, có nhiệm vụ phân loại từng câu của câu trả lời dựa trên ngữ cảnh đã cho.
                    Bạn sẽ được cung cấp một ngữ cảnh, một câu hỏi và một câu trong phần trả lời từ mô hình AI. Nhiệm vụ của Bạn là đánh giá câu trả lời đó dựa trên ngữ cảnh và câu hỏi.
                    Input:
                    Question: Câu hỏi của người dùng
                    Contexts: Một hoặc nhiều đoạn văn bản được truy xuất
                    Answer: Chỉ một câu trong đoạn văn bản LLM sinh ra
                    Bạn hãy đánh giá dựa trên các nhãn sau: 
                    supported: Nội dung câu được ngữ cảnh hỗ trợ hoặc suy ra trực tiếp.
                    unsupported: Nội dung câu không được ngữ cảnh hỗ trợ, và không thể suy ra từ đó.
                    contradictory: Nội dung câu trái ngược hoặc mâu thuẫn với ngữ cảnh.
                    no_rad: Câu không yêu cầu kiểm tra thực tế (ví dụ: câu chào hỏi, ý kiến cá nhân, câu hỏi tu từ, disclaimers).
                    Hãy trả lời bằng một trong các nhãn trên, không giải thích gì thêm. Chỉ trả lời một từ duy nhất là nhãn đó.
                    Ví dụ:
                    Question: Bạn có thể cho tôi biết về lịch sử của Câu lạc bộ Lập trình ProPTIT không?
                    Contexts: Câu lạc bộ Lập trình ProPTIT được ra đời vào năm 2011, với mục tiêu tạo ra một môi trường học tập và giao lưu cho các sinh viên đam mê lập trình.
                    Answer: Câu lạc bộ Lập trình ProPTIT được thành lập vào năm 2011.
                    Label: supported"""
            messages_groundedness += f"\n**Bây giờ hãy gán nhãn đánh giá câu trả lời dựa trên ngữ cảnh sau - Chỉ trả lời một trong các nhãn /supported/unsupported/contradictory/no_rad**\nQuestion: {query}\n\nContexts: {context}\n\nAnswer: {sentence.strip()}Label: "
            groundedness_response = llm_model.generate_response(messages_groundedness)
            groundedness_reply = groundedness_response.strip()
            print(index, groundedness_reply)
            if "supported" in groundedness_reply and "unsupported" not in groundedness_reply:
                hits += 1
                cnt += 1
            elif "unsupported" in groundedness_reply or "contradictory" in groundedness_reply:
                cnt += 1
        total_groundedness += hits / cnt if cnt > 0 else 0
    df = pd.DataFrame({"groundedness_k": [total_groundedness / len(df_train) if len(df_train) > 0 else 0]})
    df.to_csv("groundedness_k.csv")
    return total_groundedness / len(df_train) if len(df_train) > 0 else 0 

# Hàm Response Relevancy (LLM Answer - Measures relevance)


def generate_related_questions(response, embedding):
    # Sửa systemp prompt nếu muốn
    messages_related = """**Crucially, from now only use Vietnamese to response** Bạn là một trợ lý AI chuyên tạo ra các câu hỏi liên quan từ một câu trả lời. Bạn sẽ được cung cấp một câu trả lời và nhiệm vụ của bạn là tạo ra các câu hỏi liên quan đến câu trả lời đó. Hãy tạo ra ít nhất 5 câu hỏi liên quan, mỗi câu hỏi nên ngắn gọn và rõ ràng. Trả lời dưới dạng list các câu hỏi như ở ví dụ dưới. LƯU Ý: Trả lời dưới dạng ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3", ...], bao gồm cả dấu ngoặc vuông.
            Ví dụ:
            Câu trả lời: Câu lạc bộ Lập Trình PTIT (Programming PTIT), tên viết tắt là PROPTIT được thành lập ngày 9/10/2011. Với phương châm hoạt động "Chia sẻ để cùng nhau phát triển", câu lạc bộ là nơi giao lưu, đào tạo các môn lập trình và các môn học trong trường, tạo điều kiện để sinh viên trong Học viện có môi trường học tập năng động sáng tạo. Slogan: Lập Trình PTIT - Lập trình từ trái tim.
            Output của bạn: "["CLB Lập Trình PTIT được thành lập khi nào?", "Slogan của CLB là gì?", "Mục tiêu của CLB là gì?"]"
            Câu trả lời: Nếu bạn thuộc ngành khác bạn vẫn có thể tham gia CLB chúng mình. Nếu định hướng của bạn hoàn toàn là theo CNTT thì CLB chắc chắn là nơi phù hợp nhất để các bạn phát triển. Trở ngại lớn nhất sẽ là do bạn theo một hướng khác nữa nên sẽ phải tập trung vào cả 2 mảng nên sẽ cần cố gắng nhiều hơn.
            Output của bạn: "["Ngành nào có thể tham gia CLB?", "CLB phù hợp với những ai?", "Trở ngại lớn nhất khi tham gia CLB là gì?"]"""

    messages_related += f"\n**Bây giờ hãy đưa ra danh sách các câu hỏi dựa trên câu trả lời sau - LƯU Ý: Trả lời dưới dạng ['câu hỏi 1', 'câu hỏi 2', 'câu hỏi 3'], ..., bao gồm cả dấu ngoặc vuông - không giải thích gì thêm**\nCâu trả lời: {response}Output của bạn: [?]"
    # Gọi  API để tạo ra các câu hỏi liên quan
    related_response = llm_model.generate_response(messages_related)
    related_questions = related_response.strip()
    return related_questions

def response_relevancy_k(file_clb_proptit, file_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_relevancy = 0

    for index, row in df_train.iterrows():
        hits = 0
        cnt = 0
        query = row['Query']
        results = [int(id) for id in retrieval_result.iloc[index]["doc_ids"].split(",")][:k]
        reply = row['Ground truth answer']
        response = retrieval_result.iloc[index][f"response_{k}"]
        while True:
            try:    
                related_questions = generate_related_questions(response, embedding) # "["CLB Lập Trình PTIT được thành lập khi nào?", "Slogan của CLB là gì?", "Mục tiêu của CLB là gì?"]"
                related_questions = eval(related_questions) if related_questions else [] 
                break
            except Exception as e:
                print(e)
                pass
        scores =[]
        for question in related_questions:
            print(index, question)
            score = reranker_model.get_score(query, question)
            scores.append(score)
        min_score = np.min(scores)
        max_score =np.max(scores)
#         max_score = np.max(list(llm_scores.values()))
        if max_score - min_score == 0: scores = [1.0 for i in len(related_questions)]
        else: scores = (scores - min_score)/(max_score - min_score)
        hits = np.sum(scores) 
        total_relevancy += hits / len(related_questions) if len(related_questions) > 0 else 0
    df = pd.DataFrame({"response_relevancy_k": [total_relevancy / len(df_train) if len(df_train) > 0 else 0]})
    df.to_csv("response_relevancy_k.csv")
    return total_relevancy / len(df_train) if len(df_train) > 0 else 0


# Hàm Noise Sensitivity (LLM Answer - Robustness to Hallucination)

def noise_sensitivity_k(file_clb_proptit, file_train, embedding, vector_db, k=5):
    df_clb = pd.read_csv(file_clb_proptit)
    df_train = pd.read_excel(file_train)

    total_sensitivity = 0

    for index, row in df_train.iterrows():
        hits = 0
        cnt = 0
        query = row['Query']
        results = [int(id) for id in retrieval_result.iloc[index]["doc_ids"].split(",")][:k]
    
        response = retrieval_result.iloc[index][f"response_{k}"]

        sentences = response.split('. ')
        for sentence in sentences:
            # Sửa prompt nếu muốn
            messages_sensitivity = """**Crucially, from now only use Vietnamese to response** Bạn là một chuyên gia đánh giá độ nhạy cảm của câu trả lời trong hệ thống RAG, có nhiệm vụ phân loại từng câu của câu trả lời dựa trên ngữ cảnh đã cho.
                    Bạn sẽ được cung cấp một ngữ cảnh, một câu hỏi và một câu trong phần trả lời từ mô hình AI. Nhiệm vụ của là đánh giá câu trả lời đó dựa trên ngữ cảnh và câu hỏi.
                    Input:
                    Question: Câu hỏi của người dùng
                    Contexts: Một hoặc nhiều đoạn văn bản được truy xuất
                    Answer: Chỉ một câu trong đoạn văn bản LLM sinh ra
                    Bạn hãy đánh giá dựa trên các nhãn sau: 
                    1: Nội dung câu được ngữ cảnh hỗ trợ hoặc suy ra trực tiếp.
                    0: Nội dung câu không được ngữ cảnh hỗ trợ, và không thể suy ra từ đó.
                    Ví dụ:
                    Question: Bạn có thể cho tôi biết về lịch sử của Câu lạc bộ Lập trình ProPTIT không?
                    Contexts: Câu lạc bộ Lập trình ProPTIT được ra đời vào năm 2011, với mục tiêu tạo ra một môi trường học tập và giao lưu cho các sinh viên đam mê lập trình.
                    Answer: Câu lạc bộ Lập trình ProPTIT được thành lập vào năm 2011.
                    Score: 1
                    Question: Câu lạc bộ Lập trình ProPTIT được thành lập vào năm 2011. Bạn có biết ngày cụ thể không?
                    Contexts: Câu lạc bộ Lập trình ProPTIT được ra đời vào năm 2011, với mục tiêu tạo ra một môi trường học tập và giao lưu cho các sinh viên đam mê lập trình.
                    Answer: Câu lạc bộ Lập trình ProPTIT là CLB thuộc PTIT.
                    Score: 0"""

            context = "Content từ các tài liệu liên quan:\n"
            context += "\n".join([df_clb.iloc[result-1]["Văn bản"] for result in results])
            messages_sensitivity += f"\n **Bây giờ hãy gán nhãn cho câu trả lời sau - LƯU Ý chỉ gán nhãn Score 0 hoặc 1, không giải thích gì thêm**\nQuestion: {query}\n\nContexts: {context}\n\nAnswer: {sentence.strip()} Score: [?]"
            
            sensitivity_response = llm_model.generate_response(messages_sensitivity)
            sensitivity_reply = sensitivity_response.strip()
            print(index, sensitivity_reply)
            if "0" in sensitivity_reply:
                hits += 1
        total_sensitivity += hits / len(sentences) if len(sentences) > 0 else 0
    df = pd.DataFrame({"noise_sensitivity_k": [total_sensitivity / len(df_train) if len(df_train) > 0 else 0]})
    df.to_csv("noise_sensitivity_k.csv")
    return total_sensitivity / len(df_train) if len(df_train) > 0 else 0


# Hàm để tính toán toàn bộ metrics trong module LLM Answer

def calculate_metrics_llm_answer(file_clb_proptit, file_train, embedding, vector_db, train):
    # Tạo ra 1 bảng csv, cột thứ nhất là K value, các cột còn lại là metrics. Sẽ có 3 hàng tương trưng với k = 3, 5, 7
    global df_entities
    global retrieval_result
    if train:
        df_entities = pd.read_csv("train_entities.csv")
        retrieval_result = pd.read_excel("train_result.xlsx")
    else:
        df_entities = pd.read_csv("test_entities.csv")
        retrieval_result = pd.read_excel("test_result.xlsx")
    k_values = [3, 5, 7]
    metrics = {
        "K": [],
        "string_presence@k": [],
        "rouge_l@k": [],
        "bleu_4@k": [],
        "groundedness@k": [],
        "response_relevancy@k": [],
        "noise_sensitivity@k": []
    }
    # Lưu 2 chữ số thập phân cho các metrics
    for k in k_values:
        metrics["K"].append(k)
        metrics["string_presence@k"].append(round(string_presence_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["rouge_l@k"].append(round(rouge_l_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["bleu_4@k"].append(round(bleu_4_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["groundedness@k"].append(round(groundedness_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["response_relevancy@k"].append(round(response_relevancy_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics["noise_sensitivity@k"].append(round(noise_sensitivity_k(file_clb_proptit, file_train, embedding, vector_db, k), 2))
        metrics_df = pd.DataFrame(metrics)
    # Lưu DataFrame vào file csv
        if train:
            metrics_df.to_csv("metrics_llm_answer_train.csv", index=False)
        else:
            metrics_df.to_csv("metrics_llm_answer_test.csv", index=False)
        print("done---")
    metrics_df = pd.DataFrame(metrics)
    # Lưu DataFrame vào file csv
    if train:
        metrics_df.to_csv("metrics_llm_answer_train.csv", index=False)
    else:
        metrics_df.to_csv("metrics_llm_answer_test.csv", index=False)
    return metrics_df

