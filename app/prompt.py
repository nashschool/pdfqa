from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import os


class RAGSetup:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables
        self.groq_api_key = os.getenv("GROQ_KEY", "default_value_if_not_set")

    def create_rag_chain(self):
        """
        Creates a RAG chain using the provided template and input variables.
        """
        prompt = PromptTemplate(
            template=self.template, input_variables=self.input_variables
        )
        llm = ChatGroq(
            temperature=0.5,
            model_name="llama-3.3-70b-versatile",
            groq_api_key=self.groq_api_key,
        )
        rag_chain = prompt | llm | StrOutputParser()
        return rag_chain
