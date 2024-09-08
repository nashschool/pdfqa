from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import uvicorn
import logging
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from fastapi.exceptions import RequestValidationError
from .retriever import Retriever
from langchain_core.output_parsers import StrOutputParser
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logging.error(f"Validation error for {request.url}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

@app.post("/ask-question")
async def ask_question(file: UploadFile = File(...), question: str = Form(...)):
    try:
        file_content = await file.read()
        doc_splits = process_document(file_content)
        rag_chain = rag_setup()
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        retriever = Retriever(doc_splits, embeddings)

        context = retriever.query(question)
        print(f"context: {context}, question: {question}")
        generation = rag_chain.invoke({"context": context, "question": question})
        # answer_gen = gen(llm, retriever, question)
        # return JSONResponse(content={"answer": answer_gen})
        return JSONResponse(content={"answer": generation})
        

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)

def rag_setup():
    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the mathematical question. You can only make conversations based on the provided information and should refrain from making assumptions. If information isn’t available in context to answer, politely say you don’t have knowledge about that. If it is a related mathematical question, explain the answer in detail, using steps where necessary. <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question}, Context: {context}, Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "context"]
    )
    groq_api_key = os.getenv('GROQ_KEY', 'default_value_if_not_set')

    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=groq_api_key)
    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def setup_retriever(doc_splits):
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = Chroma.from_documents(documents=doc_splits, embedding=embedding)
    return vectorstore.as_retriever()


def gen(llm, retriever, input_question):
    
    print(f'ques is: {input_question}')
    retriever = Retriever(doc_splits, embeddings)

    generation = conversational_rag_chain.invoke(
        {"input": input_question},
        config={
            "configurable": {"session_id": "abc123"}
        },  
    )["answer"]
    return generation

def gen_with_history(llm, retriever, input_question):
    
    print(f'ques is: {input_question}')
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
        "Help on a clarifying question, which is based on the"
        "question and the answer that precedes it. "
        "If you think the question is irrelevant, say that you "
        "don't know. If it is mathematical, answer in steps."
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
        {"input": input_question},
        config={
            "configurable": {"session_id": "abc123"}
        },  
    )["answer"]
    return generation

def process_document(file_content):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_content)
        tmp_file.flush()
        docs = [PyPDFLoader(tmp_file.name).load()]
        docs_list = [item for sublist in docs for item in sublist]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
        return text_splitter.split_documents(docs_list)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
