from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import uvicorn
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from fastapi.exceptions import RequestValidationError
from .retriever import Retriever
from .prompt import RAGSetup
from fastapi.middleware.cors import CORSMiddleware
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


@app.post("/formula-extractor")
async def formula_extractor(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        question, correct_answer = extract_question_and_answer_from_pdf(file_content)
        formula_extractor_template = """
        You are a formula extractor. Given a question and a correct answer,
        explain the mathematical formulas used in solving the question. The formulas must be generic and the notation must be explained.
        Example Answer: Formula 1. Distance formula: \(D = sqrt((x2-x1)^2+(y2-y1)^2)\) between points \((x1,y1)\) and \((x2,y2)\). Formula 2.
        No need to allude to the specific problem and its numbers. 
        Ensure to enclose mathematical symbols and equations inside \(...\).
        Just list the formulas one by one; no intro or conclusion or additional
        explanation is necessary.
        Question:
        {question}
        Correct Answer:
        {correct_answer}"""

        formula_extractor = RAGSetup(formula_extractor_template, ["question", "correct_answer"])
        rag_chain = formula_extractor.create_rag_chain()
        feedback = rag_chain.invoke({
            "question": question,
            "correct_answer": correct_answer,
        })
        return JSONResponse(content={"answer": feedback})
    
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.post("/check-answer")
async def check_ans(file: UploadFile = File(...), user_answer: str = Form(...)):
    try:
        file_content = await file.read()
        question, correct_answer = extract_question_and_answer_from_pdf(file_content)
        prompt ="""You are an examiner who provides detailed feedback and fair marking.

        Question:
        {question}

        Correct Answer:
        {correct_answer}

        User's Answer:
        {user_answer}

        Please compare the user's answer with the correct answer. Your answer must have:
        - Marks out of 10
        - 3-line feedback addressed to the user
        For irrelevant answer, you must only say Irrelevant in the feeback. Ensure to enclose mathematical symbols 
        and equations inside \(...\)."""
       
        answer_checker = RAGSetup(prompt, ["question", "correct_answer",  "user_answer"])
        rag_chain = answer_checker.create_rag_chain()
        feedback = rag_chain.invoke({
            "question": question,
            "correct_answer": correct_answer,
            "user_answer": user_answer
        })
        return JSONResponse(content={"answer": feedback})

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/ask-question")
async def ask_question(file: UploadFile = File(...), user_question: str = Form(...), conversation_history: str = Form("")):
    try:
        file_content = await file.read()
        question, correct_answer = extract_question_and_answer_from_pdf(file_content)
        prompt = """You are an assistant for question-answering tasks. 
        You are given a question, its correct answer, and a conversation history. 
        You have to answer the user's follow-up question based on this context.

        Original Question:
        {question}

        Correct Answer:
        {correct_answer}

        Conversation History:
        {conversation_history}

        User Question: {user_question}

        Respond to the user's question, taking into account the original question, 
        its correct answer, and the conversation history. You can only use the 
        provided information and should refrain from making assumptions. If 
        information isn't available in the context to answer, politely say you 
        don't have knowledge about that. If it is a related mathematical question, 
        explain the answer in detail, using steps where necessary. Ensure to 
        enclose mathematical symbols and equations inside \(...\)."""
        
        ask_ques = RAGSetup(prompt, ["question", "correct_answer", "conversation_history", "user_question"])
        rag_chain = ask_ques.create_rag_chain()
        ai_clarification = rag_chain.invoke({
            "question": question,
            "correct_answer": correct_answer,
            "conversation_history": conversation_history,
            "user_question": user_question
        })

        return JSONResponse(content={"answer": ai_clarification})

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)

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

def extract_question_and_answer_from_pdf(file_content):
    """
    Extracts the question and correct answer from a PDF file.
    Assumes the PDF contains 'Question:' and 'Answer:' as markers.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_content)
        tmp_file.flush()
        docs = [PyPDFLoader(tmp_file.name).load()]
        text = "\n".join([doc.page_content for d in docs for doc in d])
        # Extract the question and answer
        question_start = text.find('Question:')   
        answer_start = text.find('Solution:')
        if question_start != -1 and answer_start != -1:
            question = text[question_start + len('Question:'):answer_start].strip()
            correct_answer = text[answer_start + len('Solution:'):].strip()
        else:
            raise ValueError("Could not find 'Question:' or 'Solution:' markers in the PDF.") 
        return question, correct_answer
        

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
