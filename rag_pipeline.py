import os
import shutil
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

class CustomHuggingFaceEmbeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
        
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text)
        return embedding.tolist()

class TardigradeRAG:
    def __init__(self, docs_dir="data/documents", db_dir="data/chroma_db", model_name="gpt-4o-mini"):
        self.docs_dir = docs_dir
        self.db_dir = db_dir
        self.model_name = model_name
        self.embeddings = CustomHuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self.rag_chain_ready = False

    def build_vector_store(self, force_refresh=False):
        if force_refresh and os.path.exists(self.db_dir):
            shutil.rmtree(self.db_dir)
            
        if os.path.exists(self.db_dir) and os.listdir(self.db_dir):
            # Load existing
            self.vector_store = Chroma(persist_directory=self.db_dir, embedding_function=self.embeddings)
            print("Loaded existing vector store.")
        else:
            # Build new
            print(f"Loading documents from {self.docs_dir}...")
            loader = DirectoryLoader(self.docs_dir, glob="**/*.txt", loader_cls=TextLoader)
            documents = loader.load()
            
            print("Splitting documents...")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_documents(documents)
            
            print(f"Creating vector store with {len(chunks)} chunks...")
            self.vector_store = Chroma.from_documents(
                documents=chunks, 
                embedding=self.embeddings,
                persist_directory=self.db_dir
            )
            print("Vector store created and persisted.")

    def setup_rag_chain(self):
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Call build_vector_store() first.")
            
        self.llm = ChatOpenAI(model=self.model_name, temperature=0)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        system_prompt = (
            "You are an expert on the biology and survival mechanisms of tardigrades. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use concise, accurate sentences based only on the context.\n\n"
            "{context}"
        )
        
        self.prompt = PromptTemplate(
            input_variables=["context", "input"],
            template=system_prompt + "\n\nQuestion: {input}\nAnswer:"
        )
        
        self.rag_chain_ready = True

    def query(self, question: str):
        if not self.rag_chain_ready:
            self.setup_rag_chain()
            
        docs = self.retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        formatted_prompt = self.prompt.format(context=context, input=question)
        response = self.llm.invoke(formatted_prompt)
        
        return {
            "answer": response.content,
            "source_documents": docs
        }
