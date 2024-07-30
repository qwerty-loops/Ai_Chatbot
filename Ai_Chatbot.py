import openai
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory

# Load environment variables from .env file
load_dotenv()

# Set up your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define a prompt template
prompt_template = """
You are a helpful assistant. Answer the following question:

{chat_history}
Question: {question}

Answer:
"""

# Create a LangChain prompt template
template = PromptTemplate(template=prompt_template, input_variables=["question", "chat_history"])

# Create a ChatOpenAI instance
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, verbose=True)  # Use appropriate model name

# Initialize conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

#Create a class for the AI agent and define a run method
class AIAgent:
    def __init__(self, template, llm, memory):
        self.template = template
        self.llm = llm
        self.memory = memory

    def run(self, prompt):
        # Retrieve chat history
        chat_history = self.memory.load_memory_variables({})
        formatted_prompt = self.template.format(question=prompt, chat_history=chat_history['chat_history'])
        response = self.llm.invoke(formatted_prompt)
        
        # Update memory with the latest interaction
        self.memory.save_context({"question": prompt}, {"response": response.content})
        
        return response.content

# Usage
agent = AIAgent(template, llm, conversational_memory)

while True:
    question = input("Ask your question: ")
    if question.lower() == "exit":
        print("Goodbye!")
        break
    else:
        response = agent.run(question)
        print(f"Assistant: {response}")
