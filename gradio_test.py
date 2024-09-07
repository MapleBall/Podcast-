import torch
import gradio as gr
torch.cuda.empty_cache()
gr.close_all()


from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import os


k = 5
fetch_k = 100

vectorstore_path = "/home/starklab/Desktop/Podcast/chroma_db"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 使用与创建时相同的嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={'device': device}
)

# 实例化 Chroma，并指定持久化路径
vectorstore = Chroma(
    persist_directory=vectorstore_path,
    embedding_function=embeddings
)

def setup_qa_chain():
    # vectorstore = load_vectorstore(ckpt_vectorstore)
    groq_api_key = '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'
    model = 'llama-3.1-8b-instant'
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": fetch_k})

    template = """你是一個智能助手，負責幫助使用者回答有關 Podcast 節目內容的問題。根據以下提供的檢索到的資料來回答問題。如果檢索資料中完全沒有相關信息，請直接回答「RAG 資料庫沒有您想要的資料」，否則請根據已有資料盡可能回答問題，並請使用繁體中文回答。

檢索資料信息（包括節目標題）：
{context}

聊天歷史：
{chat_history}

當前問題：
{question}

回答指南：
1. **問題處理**：首先對當前問題進行清晰的 prompt engineering，確保理解問題的核心需求。
2. **信息使用**：僅使用檢索資料中的信息來回答問題。如果資料不足以回答問題，請直接回答「RAG 資料庫沒有您想要的資料」。
3. **回答內容**：
   - **具體內容要點**：回答應包括具體的內容要點。
   - **時間戳**：每個內容要點應附上對應的時間戳（格式：MM:SS~MM:SS）。
   - **節目標題**：最後應提供節目標題（格式：（節目標題：[完整標題]））。
4. **回答格式示例**：
   - 「根據檢索資料，[內容摘要1]（時間戳）。此外，[內容摘要2]（時間戳）。[如有更多內容，繼續列舉]。（節目標題：[完整標題]）」
5. **回答語言和風格**：回答要簡潔明瞭，使用繁體中文。
6. **資訊限制**：不要添加任何檢索資料中沒有的信息。
7. **格式問題**: 請不要使用刪除線在你的回答中。
"""


    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="內容: {page_content}\n來源: {source}"
    )
    prompt = ChatPromptTemplate.from_template(template)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=groq_chat,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": prompt,
            "document_variable_name": "context",
            "document_prompt": document_prompt
        }
    )

    return qa_chain, retriever

qa_chain, retriever = setup_qa_chain()


def get_program_list(folder_path):
    try:
        # 獲取指定資料夾中的所有子資料夾名稱
        programs = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
        
        # 格式化為字符串，每行一個節目
        program_list = "\n".join(f"{i + 1}: {program}" for i, program in enumerate(programs))
        return program_list
    except FileNotFoundError:
        return "指定的資料夾不存在。"
    except Exception as e:
        return f"發生錯誤: {e}"

def chat_function(message, history):
    try:
        # 執行檢索
        results = retriever.invoke(message)
        # 根據提供的問題進行檢索和回答
        response = qa_chain.invoke({"question": message, "chat_history": history})
        answer = response['answer']

        # 構建來源信息
        sources = "\n可參考下方節目集數：\n"
        for idx, result in enumerate(results):
            filename = os.path.splitext(result.metadata['source'])[0]
            sources += f"Result {idx+1}: {filename}\n"

        return answer + "\n\n" + sources
    except Exception as e:
        return f"發生錯誤: {str(e)}\n很抱歉，我無法處理您的問題。請再試一次或換個問題。"

# 創建 Gradio 界面
with gr.Blocks() as iface:
    # 顯示節目清單
    gr.Markdown(f"## 目前資料庫中的節目有：\n{get_program_list('/media/starklab/BACKUP/Podcast_project/轉錄文本存放區')}\n\n請在下方提問：")

    # 創建聊天界面
    chatbot = gr.ChatInterface(
        chat_function,
        title="Podcast Q&A Assistant",
        description="Ask questions about podcast content, and I'll provide answers based on the retrieved information.",
        theme="soft",
        examples=[
            "林書豪這個賽季遇到了什麼困難？",
            "請告訴我這個節目討論了哪些主題？",
            "這集節目中有提到哪些重要的觀點？"
        ],
        retry_btn="重試",
        undo_btn="撤銷",
        clear_btn="清除"
    )

iface.launch()