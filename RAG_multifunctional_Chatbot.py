import openai
import os
import pandas as pd
import warnings
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.tools import PythonREPLTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS

# Ignore specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_core")

# Load environment variables from .env file
load_dotenv()

# Set up your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

#Setup Tavily API key
TavilySearchResults.api_key = os.getenv("TAVILY_API_KEY")

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

# Prompt the user to enter the file path
print("Welcome to NoteSight AI Assistant!")
file_path = input("Please enter the file path: (P.S. Enter the file path of the Excel file, without quotes) ")

# Load document
document_text = load_excel(file_path)

# Create embeddings and vector store for document retrieval
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
vector_store = FAISS.from_texts([document_text], embeddings)

# Define agent instructions
instructions = """
You are a helpful assistant. 
Your main purpose is to assist users with questions based on the provided context.
You have access to a python REPL and a search tool, which you can use to execute python code and fetch web data.
If the prompt being asked is not related to the provided context or document, search online for the information.
If you get an error, debug your code and try again.
You might know the answer without running any code, but you should still run the code to get the answer.
If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.

{chat_history}
Context from document:
{context}
Question: {question}

Answer:
"""
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)

# Create the agent
tools = [PythonREPLTool(), TavilySearchResults()]
agent = create_openai_functions_agent(ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0), tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Create a class for the AI agent and define a run method
class AIAgent:
    def __init__(self, memory, retriever, agent_executor):
        self.memory = memory
        self.retriever = retriever
        self.agent_executor = agent_executor
        self.code_output = None

    def run(self, prompt):
        # Retrieve chat history
        chat_history = self.memory.load_memory_variables({})

        # Retrieve relevant documents
        docs = self.retriever.get_relevant_documents(prompt)
        context = "\n".join([doc.page_content for doc in docs])

        # Add code output to context if available
        if self.code_output:
            context += f"\n\nCode Output:\n{self.code_output}"

        # Format the prompt with chat history and context
        formatted_prompt = instructions.format(question=prompt, context=context, chat_history=chat_history.get('chat_history', ''))
        
        response = self.agent_executor.invoke({"input": formatted_prompt})

        # Ensure the response is a string
        if isinstance(response, list):
            response = " ".join(response)
        elif not isinstance(response, str):
            response = str(response)

        # Update memory with the latest interaction
        self.memory.save_context({"question": prompt}, {"response": response})

        return response

    def run_code(self, code):
        try:
            # Execute the code using the agent executor
            result = self.agent_executor.invoke({"input": code})
            self.code_output = result
            return result
        except Exception as e:
            self.code_output = str(e)
            return str(e)

# Initialize conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=10,
    return_messages=True
)

# Initialize the retriever
retriever = vector_store.as_retriever()

# Usage
agent = AIAgent(conversational_memory, retriever, agent_executor)

while True:
    question = input("Ask your question or enter 'exit' to quit: ")
    if question.lower() == "exit":
        print("Goodbye!")
        break
    else:
        response = agent.run(question)
        # print(f"Assistant: {response}")
