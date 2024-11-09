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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "https://capeup.vercel.app",
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


@app.post("/step-evaluator")
async def step_evaluator(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        question, correct_answer, prev_answer, curr_answer = (
            extract_question_and_answer_conv_from_pdf(file_content)
        )
        step_evaluator_template = """You are an examiner who provides detailed feedback and fair marking.

        Question:
        {question}

        Correct Answer:
        {correct_answer}

        User's current step:
        {user_curr_step}

        User's previous steps:
        {user_prev_steps}

        Address the user on their current step of the answer and see if they are building on their previous steps, and by
        detailing the solution till the point of the current step, encourage them to get to the next step and the correct answer. 
        Enclose all math inside \( ... \)

        Language: The user can also enter the chat in other languages like Hindi or Hinglish or Tamil, etc, 
        in which case you must respond in the same language as the user. 
        
        Keep your guidance concise.
        If the user has the correct answer, your feedback must have:
        - Marks out of 10 (based on the current and previous steps )
        - 3-line feedback addressed to the user
        For irrelevant answer, you must only say Irrelevant in the feeback."""

        formula_extractor = RAGSetup(
            step_evaluator_template,
            ["question", "correct_answer", "user_curr_step", "user_prev_steps"],
        )
        rag_chain = formula_extractor.create_rag_chain()
        feedback = rag_chain.invoke(
            {
                "question": question,
                "correct_answer": correct_answer,
                "user_curr_step": curr_answer,
                "user_prev_steps": prev_answer,
            }
        )
        return JSONResponse(content={"answer": feedback})

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/formula-extractor")
async def formula_extractor(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        question, correct_answer = extract_question_and_answer_from_pdf(file_content)
        formula_extractor_template = """
You are a specialized formula extractor. Given a question and its correct answer, identify and list the mathematical formulas necessary to solve this question. 

**Requirements**:
1. **Formula Selection**: Do not include elementary formulas like the Pythagorean theorem or \(\sin^2(\theta) + \cos^2(\theta) = 1\).
2. **Notation Explanation**: For each formula, provide clear notation definitions.
3. **Math Formatting**: You must enclose all mathematical symbols and equations within \(...\).
4. **Response Structure**: List each formula directly, one by one, without any introductory phrases, conclusions, or extraneous explanations. Do not reference specific numbers or details from the question.

**Formatting Example**:
- Formula 1. Distance formula: \(D = sqrt((x2-x1)^2+(y2-y1)^2)\), where \(D\) is the distance between points \((x_1, y_1)\) and \((x_2, y_2)\).
- Formula 2. Sine addition formula: \(\sin(A + B) = \sin A \cos B + \cos A \sin B\).

**Task**:
Question:
{question}

Correct Answer:
{correct_answer}
"""

        formula_extractor = RAGSetup(
            formula_extractor_template, ["question", "correct_answer"]
        )
        rag_chain = formula_extractor.create_rag_chain()
        feedback = rag_chain.invoke(
            {
                "question": question,
                "correct_answer": correct_answer,
            }
        )
        return JSONResponse(content={"answer": feedback})

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/check-answer")
async def check_ans(file: UploadFile = File(...), user_answer: str = Form(...)):
    try:
        file_content = await file.read()
        question, correct_answer = extract_question_and_answer_from_pdf(file_content)
        prompt = """
You are an examiner responsible for providing detailed feedback and fair marking of the user's answer.

### Task:
Evaluate the user's answer by comparing it to the correct answer provided. 

**Guidelines**:
1. **Scoring**: Assign a score out of 10 based on the accuracy and completeness of the user's answer.
2. **Feedback**:
   - Write exactly 3 lines of constructive, encouraging feedback addressed directly to the user.
   - If the answer is irrelevant, clearly state "Irrelevant" and politely encourage the user to refocus on their studies.
3. **Language Matching**: Respond in the same language as the user's answer (e.g., Hindi, Hinglish, Tamil).
4. **Mathematical Formatting**: You must enclose all mathematical symbols and equations inside \(...\) to ensure clear presentation.

### Response Format:
1. **Marks**: /10
2. **Feedback**: 
   - Line 1:
   - Line 2:
   - Line 3:

### Evaluation Context:
- **Question**: {question}
- **Correct Answer**: {correct_answer}
- **User's Answer**: {user_answer}
"""

        answer_checker = RAGSetup(prompt, ["question", "correct_answer", "user_answer"])
        rag_chain = answer_checker.create_rag_chain()
        feedback = rag_chain.invoke(
            {
                "question": question,
                "correct_answer": correct_answer,
                "user_answer": user_answer,
            }
        )
        return JSONResponse(content={"answer": feedback})

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/ask-question")
async def ask_question(
    file: UploadFile = File(...),
    user_question: str = Form(...),
    conversation_history: str = Form(""),
):
    try:
        file_content = await file.read()
        question, correct_answer = extract_question_and_answer_from_pdf(file_content)
        prompt = """
You are a highly specialized assistant dedicated to answering only relevant questions in a question-answering task. Your responses must strictly adhere to the instructions below.

Context Provided:
- **Original Question:** {question}
- **Correct Answer:** {correct_answer}
- **Conversation History:** {conversation_history}
- **User's Follow-up Question:** {user_question}

### Instructions:
1. **Response Scope**: Answer only the user's follow-up question, using information from the original question, the correct answer, and the conversation history. Do not introduce any information beyond what is provided or known.
   
2. **Language Matching**: Respond in the same language used by the user, such as Hindi, Hinglish, Tamil, etc. If unsure, default to the language used in the user's question.

3. **Mathematical Clarity**: For math-related questions, provide a detailed explanation. Present each step clearly and enclose all mathematical symbols and equations in LaTeX format, using \(...\) for inline expressions. Do not use $...$ or $$...$$

4. **Assumptions**: Do not make assumptions. Answer only based on the provided context or general knowledge applicable to the question. If the context lacks information for a complete answer, state only what is verifiable without guessing.

5. **Relevance Check**: If the user's question or comment is irrelevant, politely decline to answer. Encourage the user to focus on relevant study topics.

6. **Politeness and Encouragement**: Always be polite and encouraging. Politely guide the user to ask relevant questions if they deviate.

### Important:
Failure to follow these instructions strictly will result in unsatisfactory assistance to the user. Ensure each response aligns exactly with these guidelines.
"""

        ask_ques = RAGSetup(
            prompt,
            ["question", "correct_answer", "conversation_history", "user_question"],
        )
        rag_chain = ask_ques.create_rag_chain()
        ai_clarification = rag_chain.invoke(
            {
                "question": question,
                "correct_answer": correct_answer,
                "conversation_history": conversation_history,
                "user_question": user_question,
            }
        )

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

    print(f"ques is: {input_question}")
    retriever = Retriever(doc_splits, embeddings)

    generation = conversational_rag_chain.invoke(
        {"input": input_question},
        config={"configurable": {"session_id": "abc123"}},
    )["answer"]
    return generation


def gen_with_history(llm, retriever, input_question):

    print(f"ques is: {input_question}")
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
        config={"configurable": {"session_id": "abc123"}},
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
        question_start = text.find("Question:")
        answer_start = text.find("Solution:")
        if question_start != -1 and answer_start != -1:
            question = text[question_start + len("Question:") : answer_start].strip()
            correct_answer = text[answer_start + len("Solution:") :].strip()
        else:
            raise ValueError(
                "Could not find 'Question:' or 'Solution:' markers in the PDF."
            )
        return question, correct_answer


def extract_question_and_answer_conv_from_pdf(file_content):
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
        question_start = text.find("Question:")
        answer_start = text.find("Solution:")
        prev_step = text.find("Prev Step:")
        curr_step = text.find("Current Step:")
        if question_start != -1 and answer_start != -1:
            question = text[question_start + len("Question:") : answer_start].strip()
            correct_answer = text[answer_start + len("Solution:") :].strip()
        else:
            raise ValueError(
                "Could not find 'Question:' or 'Solution:' markers in the PDF."
            )
        return question, correct_answer, prev_step, curr_step


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
