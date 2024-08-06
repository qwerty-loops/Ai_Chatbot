import openai
import os
import pandas as pd
import warnings
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain.schema import HumanMessage

# Ignore specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

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

# Function to load document based on file extension
def load_document(file_path):
    if file_path.endswith('.xlsx'):
        return load_excel(file_path)
    else:
        raise ValueError("Unsupported file type. Please provide an Excel file.")

# Prompt the user to enter the file path
print("Welcome to NoteSight AI Assistant!")

while True:
    try:
        file_path = input("Please enter the file path: ").strip('\'"')
        document_text = load_document(file_path)
        break
    except ValueError as e:
        print(e)

# Create embeddings and vector store for document retrieval
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
vector_store = FAISS.from_texts([document_text], embeddings)

# Define a prompt template
prompt_template = """
You are a helpful assistant. 
If you have any questions, feel free to ask.
If you are unsure of the answer, you can say "I don't know".
Answer the following question based on the provided context, which shall be an uploaded excel document.:

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

# Initialize the retriever
retriever = vector_store.as_retriever()

# Usage
agent = AIAgent(template, llm, conversational_memory, retriever)

while True:
    question = input("Ask your question or input 'exit': ")
    if question.lower() == "exit":
        print("Goodbye!")
        break
    else:
        response = agent.run(question)
        print(f"Assistant: {response}")
