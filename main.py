from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


loader = TextLoader("speech.txt", encoding='utf-8')

documents = loader.load()


splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(chunks, embeddings)
prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
""")
llm = OllamaLLM(model="mistral")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser


retriever = vectorstore.as_retriever()


retrival_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


print("\n🟢 Ambedkar Q&A System Ready!")
print("Type your question below. Type 'exit' to quit.\n")

while True:
    user_question = input("❓ Your Question: ")

    if user_question.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break

    answer = retrival_chain.invoke(user_question)
    print(f"\n🟩 Answer: {answer}\n")
