from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import tempfile
import uvicorn
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.llms import OpenAI
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from fastapi.responses import JSONResponse
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI()


def setup_retriever(doc_splits):
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = Chroma.from_documents(documents=doc_splits, embedding=embedding)
    retriever = vectorstore.as_retriever()
    return retriever

def gen(llm, retriever):
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    generation = conversational_rag_chain.invoke(
        {"input": "What is the question about?"},
        config={
            "configurable": {"session_id": "abc123"}
        },  
    )["answer"]
    return generation

def process_document(file_content):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_content)
        tmp_file_path = tmp_file.name

    docs = [PyPDFLoader(tmp_file_path).load()]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    
    return doc_splits


@app.post("/ask-question")
async def ask_question(file: UploadFile = File(...), question: str = Form(...)):
    try:
        file_content = await file.read()
        doc_splits = process_document(file_content)
        llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key='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')
        retriever = setup_retriever(doc_splits);
        answer_gen = gen(llm, retriever)
        return JSONResponse(content={"answer": answer_gen})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    # Run the application with Uvicorn on localhost port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
