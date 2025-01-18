import chromadb
from torch.nn.functional import embedding

from app.models.pdf_document_content import PdfDocumentContentModel
from app.models.response import ResponseModel
from dotenv import load_dotenv
import os

load_dotenv()

class DocumentRepository:
    """
       A repository class for handling operations related to documents in a vector database.
    """
    def __init__(self, collection_name:str):
        self.collection_name = collection_name
        self.client= chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH"))

    def store_pdf_document_in_vector_chroma(self, pdf_document_content: PdfDocumentContentModel) -> ResponseModel:
        """
            Stores the embeddings for a document.
            :param pdf_document_content:
        """
        # Initialize lists for storage
        documents = []  # This will store individual text chunks
        embeddings = []  # This will store individual embedding chunks
        metadatas = []  # This will store the metadata for each chunk
        ids = []  # This will store the IDs for each chunk

        for page_content in pdf_document_content.page_content:
            # Extract fields for each chunk in the page
            for i, (text_chunk, embedding_chunk) in enumerate(zip(page_content.text, page_content.embedding)):
                documents.append(text_chunk)  # Add individual text chunk
                embeddings.append(embedding_chunk)  # Add corresponding embedding chunk
                # Add metadata, include additional details like chunk index if needed
                metadata = page_content.page_metadata.model_dump()
                metadata.update({"chunk_index": i})  # Add chunk-specific metadata
                metadatas.append(metadata)
                ids.append(f"{page_content.id}_chunk_{i}")  # Unique ID per chunk

        # Store to db
        self.client.get_or_create_collection(self.collection_name).add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        return ResponseModel(
            status=200,
            message="Document successfully stored",
            data=None
        )

    def get_collection(self):
        collection = self.client.get_collection(self.collection_name)
        data = collection.get(include=["embeddings", "documents", "metadatas"])

        document = data["documents"]
        embeddings = data["embeddings"]
        metadatas = data["metadatas"]
        ids = data["ids"]
        content = {
            "document": document,
            "metadatas": metadatas,
            "ids": ids,
            "embeddings":embeddings
        }

        return content

    def query_vector_chroma_db(self, embedded_query: list[float]):
        """
            Queries the vector database with an embedded query.

            Args:
                embedded_query (list[float]): The embedded query to search for.

            Returns:
                dict: The query results including documents, metadatas, and distances.
        """
        return self.client.get_collection(self.collection_name).query(
            query_embeddings=embedded_query,
            n_results= 3,
            include=["documents", "metadatas", "distances"]
        )

    def peek_collection(self):
        return self.client.get_collection(name=self.collection_name).peek()

    def count_collection(self):
        return self.client.get_collection(name=self.collection_name).count()

    def change_collection_name(self, new_name:str):
        self.client.get_collection(self.collection_name).modify(name=new_name)


