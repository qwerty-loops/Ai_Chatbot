import openai
import os
import pandas as pd
import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain.schema import HumanMessage

# Ignore specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_core")

# Load environment variables from .env file
load_dotenv()

# Set up your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to load and convert Excel to text
def load_excel(file_path):
    workbook = pd.ExcelFile(file_path)
    text = ""
    for sheet_name in workbook.sheet_names:
        try:
            sheet = pd.read_excel(workbook, sheet_name=sheet_name)
            text += f"Sheet name: {sheet_name}\n"
            text += sheet.to_string(index=False, header=True)
            text += "\n\n"
        except Exception as e:
            text += f"Error reading {sheet_name}: {e}\n\n"
    return text

# Hardcoded file path
file_path = r"D:\Allen Archive\Allen Archives\NEU_academics\the_Internship_trials\Intern_stuff\Dev Work\RAG_LLM_Trials\Duane Gafoor GRADED-9.xlsx"

# Load document
document_text = load_excel(file_path)

# Create embeddings and vector store for document retrieval
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
vector_store = FAISS.from_texts([document_text], embeddings)

# Define a prompt template
prompt_template = """
You are a helpful assistant. 
Your main purpose is to assist users with questions based on the provided context.
If you have any questions, feel free to ask.
If you are unsure of the answer, you can say "I don't know".

{chat_history}
{context}
Question: {question}

Answer:
"""

# Create a LangChain prompt template
template = PromptTemplate(template=prompt_template, input_variables=["question", "chat_history", "context"])

# Create a ChatOpenAI instance
llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0, verbose=True)

# Initialize conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=7,
    return_messages=True
)

# Create a class for the AI agent and define a run method
class AIAgent:
    def __init__(self, template, llm, memory, retriever):
        self.template = template
        self.llm = llm
        self.memory = memory
        self.retriever = retriever

    def run(self, prompt):
        # Retrieve chat history
        chat_history = self.memory.load_memory_variables({})

        # Retrieve relevant documents
        docs = self.retriever.get_relevant_documents(prompt)
        context = "\n".join([doc.page_content for doc in docs])

        # Format the prompt with chat history and context
        formatted_prompt = self.template.format(question=prompt, chat_history=chat_history.get('chat_history', ''), context=context)
        messages = [HumanMessage(content=formatted_prompt)]
        
        response = self.llm(messages)

        # Update memory with the latest interaction
        self.memory.save_context({"question": prompt}, {"response": response.content})

        return response.content

    def run_code(self, code):
        try:
            # Execute the code and capture the output
            local_vars = {}
            exec(code, {}, local_vars)
            return local_vars
        except Exception as e:
            return str(e)

# Initialize the retriever
retriever = vector_store.as_retriever()

# Usage
agent = AIAgent(template, llm, conversational_memory, retriever)

while True:
    question = input("Ask your question or enter 'code' to execute code: ")
    if question.lower() == "exit":
        print("Goodbye!")
        break
    elif question.lower() == "code":
        print("Enter your Python code (end with 'END'):")
        code_lines = []
        while True:
            line = input()
            if line == "END":
                break
            code_lines.append(line)
        code = "\n".join(code_lines)
        result = agent.run_code(code)
        print(f"Result: {result}")
    else:
        response = agent.run(question)
        print(f"Assistant: {response}")
