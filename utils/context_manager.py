import os 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from setup.models import engine, veiculos
from sqlalchemy import select, inspect


class ContextManager:
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", vector_db_path: str = "faiss_index"):
        
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={'device': 'cpu'})
        self.vector_db_path = vector_db_path
        self.vector_store = None
        self.chat_history = []
        self._load_or_create_vector_store()

    def _load_or_create_vector_store(self):

        faiss_file_path = os.path.join(self.vector_db_path, "index.faiss")

        if os.path.exists(self.vector_db_path) and os.path.exists(faiss_file_path):
            print(f"Carregando FAISS index de {self.vector_db_path}...")
            try:
                self.vector_store = FAISS.load_local(self.vector_db_path, self.embeddings, allow_dangerous_deserialization=True)
                print("FAISS index carregado com sucesso.")
                return
            except Exception as e:
                print(f"Erro ao carregar FAISS index, talvez corrompido ou incompatível: {e}")
                print("Tentando recriar o index.")
                import shutil
                shutil.rmtree(self.vector_db_path)
        
        print("Nenhum FAISS index encontrado. Um novo será criado ao adicionar documentos.")
        self.vector_store = None

    def load_documents(self, directory_path: str):

        documents = []
        for filename in os.listdir(directory_path):
            filepath = os.path.join(directory_path, filename)
            try:
                if filename.endswith(".txt"):
                    loader = TextLoader(filepath)
                    documents.extend(loader.load())
                elif filepath.endswith(".pdf"):
                    loader = PyMuPDFLoader(filepath)
                    documents.extend(loader.load())
                else:
                    print(f"Ignorando arquivo não suportado: {filename}")
            except Exception as e:
                print(f"Erro ao carregar o arquivo {filename}: {e}")

        inspector = inspect(veiculos)

        column_name = [column.key for column in inspector.mapper.columns]
        
        result = engine.connect().execute(select(veiculos)).all()

        for row in result:
            db_document = Document(
                page_content = str(dict(zip(column_name,row)))    
            ) 
            documents.append(db_document)

        if not documents:
            print("Nenhum documento válido encontrado ou carregado.")
            return
        
        text_spliter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True,
        )

        chunks = text_spliter.split_documents(documents)
        print(f"Documentos processados em {len(chunks)} chuncks.")

        if os.listdir(directory_path):
            faiss_add = chunks
        else:
            faiss_add = documents

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(faiss_add,self.embeddings)
            print("Novo FAISS index criado.")
        else:
            self.vector_store.add_documents(faiss_add)
            print("Chuncks adicionados ao FAISS index existente.")

        self.vector_store.save_local(self.vector_db_path)
        print(f"FAISS index salvo em {self.vector_db_path}")

    def retrieve_context(self, query: str, k: int = 3) -> str:

        if self.vector_store is None:
            return "Nenhum documento carregado para buscar contexto"
        
        docs = self.vector_store.similarity_search(query, k=k)
        context_text = "\n\n".join([doc.page_content for doc in docs])
        print(f"\n--- Contexto recuperado ({len(docs)} chuncks) ---")

        for i, doc in enumerate(docs):
            print(f"Chunck {i+1}: {doc.page_content[:150]}...")
        print("---------------------------------------")
        return context_text
    
    def add_message_to_history(self, content: str, is_user: bool = True):

        if is_user:
            self.chat_history.append(HumanMessage(content=content))
        else:
            self.chat_history.append(AIMessage(content=content))

    def get_chat_history_for_llm(self) -> list:
        if len(self.chat_history) > 10:
            return [SystemMessage(content="Continuando a conversa...")] + self.chat_history[-10:]
        return self.chat_history