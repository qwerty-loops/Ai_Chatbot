import openai
import os
import pandas as pd
import warnings
import fitz  # For PDF parsing
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

# Function to load and convert PDF to text
def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

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

# Function to split text into smaller chunks
def split_text(text, max_length=1000):
    words = text.split()
    chunks = []
    chunk = []
    for word in words:
        chunk.append(word)
        if len(" ".join(chunk)) > max_length:
            chunks.append(" ".join(chunk))
            chunk = []
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

# Function to load document based on file extension
def load_document(file_path):
    if file_path.endswith('.pdf'):
        return load_pdf(file_path)
    elif file_path.endswith('.xlsx'):
        return load_excel(file_path)
    else:
        raise ValueError("Unsupported file type. Please provide a PDF or Excel file.")

# Prompt the user to enter the file path
print("Welcome to NoteSight AI Assistant!")
file_path = input("Please enter the file path: (P.S. Enter the file path of the PDF or Excel file, without quotes) ")

# Load document
document_text = load_document(file_path)

# Split document into smaller chunks
document_chunks = split_text(document_text)

# Create embeddings and vector store for document retrieval
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
vector_store = FAISS.from_texts(document_chunks, embeddings)

# Define a prompt template
prompt_template = """
You are a helpful assistant. 
If you have any questions, feel free to ask.
If you are unsure of the answer, you can say "I don't know".
Answer the following question based on the provided context, which shall be an uploaded pdf or excel document.:

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
