from pyexpat import model
from openai import OpenAI
from google import genai
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv
import pandas as pd
import random
import torch
import json
import dotenv
import pandas as pd

class LLM:
    def __init__(self, model_name, data_path):
        self.model_name = model_name
        self.df_base = pd.read_csv(data_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
    def generate_rank_prompt(self, query, doc):
        return f"""
        You are an information retrieval assistant for a programming club. Your task is to evaluate the relevance of documents to user questions about the club's activities, providing both a detailed reasoning and a score.

        **When providing 'Reasoning', consider the following:**
        *   Does the Document directly answer any part of the User Question?
        *   Does it address the core topic of the User Question?
        *   Does it provide specific, relevant details or general information related to the topic?
        *   **Crucially, can the answer to the User Question be reasonably inferred or deduced from the information provided in the Document, even if not explicitly stated? If a logical inference can be made to answer the question, consider this highly relevant.**
        *   **If the Document outlines general rules, obligations, or expected conduct (e.g., "adhere to club rules", "complete tasks on time"), this shows the existence of such regulations and is considered moderately to highly relevant, even if the specific details (like 'late arrival' or 'leave requests') are not mentioned.**
        *   **Distinguish clearly between the main 'User Question' and any sample Q&A pairs within the 'Document'. Focus your reasoning on the overall relevance of the Document to the main 'User Question'.**

        **Assign the 'Score' based on the 'Reasoning':**
        *   **High score (8-10):** Document directly and fully answers the User Question, provides highly specific and relevant details, OR the answer can be logically inferred with high confidence from the provided information.
        *   **Moderate score (5-7):** Document addresses the general topic of the question, provides related information (like general obligations or conduct rules), or answers a part of the question, even if not exhaustively or directly inferable.
        *   **Low score (0-4):** Document is irrelevant, tangential, or addresses a completely different topic entirely.
                
        Here are a few examples to guide your evaluation:
        ----
        User Question:
        "Tiêu chí đánh giá trong giai đoạn training là gì, và nếu em chưa giỏi lập trình thì em có thể tham gia câu lạc bộ được không ?"
        Document:
        "Trong vòng training, các anh chị sẽ đánh giá em về nhiều mặt khác nhau, bao gồm cả mảng học tập, hoạt động và cách giao tiếp giữa em với các thành viên CLB khác. Việc code chỉ là 1 phần trong số đó, em cố gắng thể hiện hết mình là được nhé, mọi nỗ lực em làm đều sẽ được anh chị ghi nhận và đánh giá. Anh chị đánh giá rất cao sự tiến bộ của các em trong quá trình training."
        Evaluation:
        "reasoning": "The user asks about training evaluation criteria and joining with weak coding skills. 
        The document directly and comprehensively addresses both parts  of the question, detailing evaluation aspects like learning, activities, and communication, and reassuring that coding skill is not the sole determinant for participation. The tone is friendly and encouraging, matching the desired style. This is highly relevant.",
        "score": 10
        ---
        User Question:
        "CLB này là CLB gì ?"
        Document:
        "Tại sao CLB lại chỉ tuyển sinh viên năm nhất ? Các sinh viên lớn hơn năm nhất có sự chênh lệch về kiến thức, thời gian hoạt động. Vì vậy việc định hướng và hoạt động sẽ không phù hợp với lộ trình đào tạo của chúng mình."
        Evaluation
        "reasoning": "The user asks 'What is the club?'. The document explains the club's recruitment policy regarding student year level, but it does not introduce the club itself (name, purpose, etc.). This information is completely irrelevant to the user's question about introducing the club.",
        "score": 0
        ---
        User Question:
        "Tiêu chí đánh giá trong giai đoạn training là gì, và nếu em chưa giỏi lập trình thì em có thể tham gia câu lạc bộ được không ?"
        Document:
        "Nhận được quyền lợi gì khi tham gia vào CLB ? Khi bạn vào CLB bạn sẽ có những lợi thế về các môn chuyên ngành trên trường. CLB sẽ giúp các bạn định hướng phát triển các kỹ năng lập trình. Bên cạnh đó CLB còn tổ chức các hoạt động ngoại khóa để các thành viên gắn kết hơn. Ngoài ra khi tham gia vào các CLB bạn sẽ được cộng 0.1 điểm khuyến khích học bổng (Đây sẽ là 1 lợi thế rất lớn khi xét học bổng tại Học Viện vào cuối kì bạn nhé)."
        Evaluation:
        "reasoning": "The user asks about 'training evaluation criteria' and 'joining with weak coding skills'. This document discusses 'benefits of joining the club' (career advantages, skill development, social activities, scholarship points). It does not address the training evaluation or joining criteria at all, making it irrelevant to the user's specific query.",
        "score": 0
        ---
        User Question:
        "CLB có mấy team dự án ạ ?"
        Document:
        "TEAM DỰ ÁN: Team AI, Team Mobile, Team Data, Team Game, Team Web, Team Backend"
        Evaluation:
        "reasoning": "The user asks 'How many project teams are there?'. The document directly lists the names of the project teams. This is a complete and direct answer to the question, making it highly relevant.",
        "score": 10
        ---
        User Question:
        "CLB tuyển thành viên như thế nào ạ ? Có bao nhiêu vòng ạ ?"
        Document:
        "Anh ơi, anh có thể mô tả rõ cho em về môi trường hoạt động trong CLB như nào không ạ ? Môi trường làm việc của CLB rất năng động, chuyên nghiệp, và liên tục đổi mới nha bạn. Bạn sẽ được gặp mặt và quen biết với nhiều bạn bè mới, được nói chuyện và học tập cùng các bạn và anh chị đều là những người rất tài năng trong trường. Ngoài ra, CLB hoạt động theo hình thức khoá trên dạy khoá dưới, các khoá sau khi ra trường đều sẽ quay lại đóng góp cho CLB."
        Evaluation:
        "reasoning": "The user asks about the 'recruitment process and number of rounds'. This document describes the 'club's working environment' (dynamic, professional, meeting new people, learning from talented members). This is completely unrelated to the recruitment process. Therefore, it is irrelevant.",
        "score": 1
        ---
        User Question:
        "Em ko học CNTT thì có tham gia được CLB ko ạ ?
        Document:
        "Chỉ cần em có đam mê với lập trình và lòng nhiệt huyết, sự cố gắng là hoàn toàn có thể nhé."
        Evaluation:
        "reasoning": "The user asks if they can join without studying CS. The document directly answers 'yes, it's possible' if they have passion, enthusiasm, and effort. This is a direct and highly relevant answer with an encouraging tone.",
        "score": 9
        ---
        User Question:
        "Nội quy của CLB về việc đi muộn và xin nghỉ như thế nào?"
        Document:
        "Nghĩa vụ: Tham gia đầy đủ, nhiệt tình, có tinh thần trách nhiệm cao trong các hoạt động của CLB. Chấp hành nghiêm chỉnh nội quy của CLB đề ra. Hoàn thành nhiệm vụ được giao đúng thời hạn quy định. Hoàn thành đóng phí hoạt động hàng tháng đúng thời gian quy định. Tham gia đóng góp ý kiến, xây dựng và phát triển CLB."
        Evaluation
        "reasoning": "The user is asking for specific rules regarding 'late arrival' and 'leave requests'. While the document does not mention these exact scenarios, it clearly states the general obligation to 'adhere strictly to the club's rules' and 'complete tasks on time'. This implies the existence of club regulations and expected member conduct. From these general obligations, it can be reasonably inferred that the club has formal rules for conduct, including procedures for absence or lateness, and that members are expected to follow them. Therefore, it's moderately relevant as it provides context for the general framework of club rules, allowing for a reasonable inference that specific procedures for absence or lateness likely exist.",
        "score": 7
        ----
        Now, evaluate the following:
        **Crucially**: Just give reasoning and score in json format - MUST BE CORRECTLY JSON FORMAT - DO NOT include any additional text, explanations or punctuation near score.
        User Question:
        {query}
        Document:
        {doc}
        Evaluation: 
        "reasoning": 
        "score":
        """
    def generate_answer_prompt(self, query, doc_ids):
        docs =""
        for i in range(len(doc_ids)):
            docs += f"Document {i+1}:\n{self.df_base.iloc[doc_ids[i] - 1]['Văn bản']}\n\n"
        return f"""
        You are a helpful and knowledgeable assistant for a programming club. Your task is to answer user questions about the club's activities, membership, training, and operations in a friendly, informative, and encouraging tone, mirroring the style of the provided examples.

        **Your response MUST be a direct answer to the User Question, using the same friendly and detailed tone as the examples.**

        **When answering, consider the following:**
        *   Directly address the User Question.
        *   Provide specific details as shown in the examples.
        *   Maintain a helpful, encouraging, and approachable tone.
        *   If the document contains information related to Q&A pairs, ensure your answer is based on the *User Question* and the most relevant information from the document.

        Here are a few examples to guide your response style:

        ---
        User Question:
        "Tiêu chí đánh giá trong giai đoạn training là gì, và nếu em chưa giỏi lập trình thì em có thể tham gia câu lạc bộ được không ?"

        Document:
        "Trong vòng training, các anh chị sẽ đánh giá em về nhiều mặt khác nhau, bao gồm cả mảng học tập, hoạt động và cách giao tiếp giữa em với các thành viên CLB khác. Việc code chỉ là 1 phần trong số đó, em cố gắng thể hiện hết mình là được nhé, mọi nỗ lực em làm đều sẽ được anh chị ghi nhận và đánh giá. Anh chị đánh giá rất cao sự tiến bộ của các em trong quá trình training."

        Answer:
        "Chào em, trong vòng training, các anh chị sẽ đánh giá em về nhiều mặt khác nhau, bao gồm cả mảng học tập, hoạt động và cách giao tiếp giữa em với các thành viên CLB khác. Việc code chỉ là một phần trong số đó thôi, quan trọng là em cố gắng thể hiện hết mình. Mọi nỗ lực của em đều sẽ được anh chị ghi nhận và đánh giá cao. CLB rất mong chờ sự tiến bộ của các em trong quá trình này nhé!"
        ---
        User Question:
        "Thành viên CLB có thể đảm nhận những vị trí lãnh đạo nào?"

        Document:
        "Thành viên có quyền ứng cử, đề cử và bầu cử vào các vị trí lãnh đạo như Chủ nhiệm, Phó chủ nhiệm, Trưởng ban đào tạo, Trưởng ban sự kiện, Trưởng ban truyền thông, v.v."

        Answer:
        "Chào em, thành viên CLB của chúng mình có thể đảm nhận nhiều vị trí lãnh đạo khác nhau đó. Em có quyền ứng cử, đề cử và bầu cử vào các vị trí như Chủ nhiệm, Phó chủ nhiệm, Trưởng ban đào tạo, Trưởng ban sự kiện, Trưởng ban truyền thông, và nhiều vị trí khác nữa nhé!"
        ---
        User Question:
        "Khi tham gia CLB, thành viên sẽ được hưởng những quyền lợi gì và cần thực hiện những nghĩa vụ gì?"

        Document:
        "Quyền lợi gồm tham gia hoạt động học tập, dự án, ứng cử – đề cử, và học hỏi kỹ năng. Nghĩa vụ gồm tham gia đầy đủ, chấp hành nội quy, hoàn thành nhiệm vụ, đóng phí đúng hạn và đóng góp ý kiến xây dựng CLB."

        Answer:
        "Khi tham gia CLB, em sẽ có rất nhiều quyền lợi hấp dẫn như được tham gia các hoạt động học tập, dự án, có cơ hội ứng cử - đề cử vào các vị trí lãnh đạo, và được học hỏi thêm nhiều kỹ năng mới. Bên cạnh đó, để CLB ngày càng phát triển, em cũng cần thực hiện một số nghĩa vụ như tham gia đầy đủ các buổi sinh hoạt, chấp hành nội quy, hoàn thành tốt nhiệm vụ được giao, đóng phí đúng hạn và tích cực đóng góp ý kiến xây dựng CLB nhé!"
        ---
        User Question:
        "CLB có hoạt động mentoring cho thành viên mới không?"

        Document:
        "Có, mentor sẽ hướng dẫn kỹ thuật, giải đáp thắc mắc và giúp thành viên mới làm quen với dự án."

        Answer:
        "Chào em, CLB có hoạt động mentoring rất chu đáo cho thành viên mới đó! Mentor của CLB sẽ hướng dẫn em về kỹ thuật, giải đáp mọi thắc mắc và giúp em làm quen với các dự án một cách nhanh chóng nhất."
        ---

        Now, evaluate the following and provide the answer in a same tone with given examples:

        User Question:
        {query}

        Document:
        {docs}

        Answer:
        """
    def generate_response(self, prompt):
            messages = [
                {"role": "assistant", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=32768
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            return content
    def get_score(self, query, doc):
        while True:
            prompt = self.generate_rank_prompt(query, doc)
            response = self.generate_response(prompt)
            try:
                start_index = response.find('{')
                end_index = response.find('}')
                json_string = response[start_index: end_index + 1]
                score_string = json.loads(json_string)["score"]
                return score_string
            except Exception as e:
                pass

if __name__ == "__main__":
    model_name = "Qwen/Qwen3-0.6B"
    df_test  = pd.read_excel("train_data_proptit.xlsx")
    doc_path = "CLB_PROPTIT.csv"
    df_doc = pd.read_csv(doc_path)
    # num_test_query = 1
    model = LLM(model_name, doc_path)
    promp = [{"role": "user", "content": "hi"}, {"role": "assistant", "content":"hello"}]
    print(model.generate_response(promp))
    # for i in range(30, 30 + num_test_query):
    #     query = df_test.iloc[i]['Query']
    #     docs = df_test.iloc[i]['Ground truth document']
    #     docs_id = []
    #     if type(docs) == str:
    #         docs_id = [int(x) for x in docs.split(',')]
    #     else:
    #         docs_id = [int(docs)]
    #     # answer = model.generate_response(model.generate_answer_prompt(query,  docs_id))
    #     # print(answer)
    #     for doc_id in docs_id:
    #         fail_id = random.randint(1, len(df_doc))
    #         while fail_id in docs_id:
    #             fail_id = random.randint(1, len(df_doc))
    #         doc = df_doc.iloc[doc_id - 1]['Văn bản']
    #         # print(doc)
    #         # print(model.generate_rank_prompt(query, doc))
    #         # response = model.generate_response(model.generate_rank_prompt(query, doc))
    #         # print(response, type(response))
    #         score = model.get_score(query, doc)
    #         print(f"{query} - {doc} - {score} -{type(score)}")
    #         print("-" * 50)
