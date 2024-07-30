import openai
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import fitz  # PyMuPDF

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
    return [text]

# Load PDF document
pdf_path = "D:\\Allen Archive\\Allen Archives\\NEU_academics\\the_Internship_trials\\Intern_stuff\\Dev Work\\RAG_LLM_Trials\\Ai_Chatbot\\cs6220_syllabus_spring2024.pdf"
documents = load_pdf(pdf_path)

# Create embeddings and vector store for document retrieval
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
vector_store = FAISS.from_texts(documents, embeddings)

# Define a prompt template
prompt_template = """
You are a helpful assistant. Answer the following question based on the provided context:

{chat_history}
{context}
Question: {question}

Answer:
"""

# Create a LangChain prompt template
template = PromptTemplate(template=prompt_template, input_variables=["question", "chat_history", "context"])

# Create a ChatOpenAI instance
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, verbose=True)  # Use appropriate model name

# Initialize conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
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
        formatted_prompt = self.template.format(question=prompt, chat_history=chat_history['chat_history'], context=context)
        response = self.llm.invoke(formatted_prompt)

        # Update memory with the latest interaction
        self.memory.save_context({"question": prompt}, {"response": response.content})

        return response.content

# Initialize the retriever
retriever = vector_store.as_retriever()

# Usage
agent = AIAgent(template, llm, conversational_memory, retriever)

while True:
    question = input("Ask your question: ")
    if question.lower() == "exit":
        print("Goodbye!")
        break
    else:
        response = agent.run(question)
        print(f"Assistant: {response}")
