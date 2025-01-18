from typing import Union, List
from dotenv import load_dotenv
import torch.cuda
from sentence_transformers import SentenceTransformer
import os
from langchain_google_genai import GoogleGenerativeAI as genAi, ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from torch import Tensor
from torch._numpy import ndarray

from app.models.ai_response import AIResponseModel

load_dotenv()

class AIService:

    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.tokenizer=None
        self.api_key=os.getenv("GOOGLE_API_KEY")
        self.model_path=os.getenv("HUGGINGFACE_MODEL_PATH")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(model_name_or_path=model_name, device=device)

    def prompt_gemini_flash(self, prompt:str, context:str):
        llm = ChatGoogleGenerativeAI(
            model = "gemini-1.5-flash",
            temperature=0,
            timeout=None,
            max_retries=2,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE
            }
        )

        prompt_template = (f"Based on the following context items, please answer the query.\n"
                       f"Give yourself room to think by extracting relevant passages from the context before answering the query.\n"
                       f"Make sure your answers are as explanatory as possible\n"
                       f"Also, give sources to your answer, you can use the page number given in teh context, as well as any links you may come across the context\n"
                       f"Now here are the context items you may use to answer the user query:\n"
                       f"context:\n{context}\n"
                       f"question: {prompt}\n"
                       f"In addition, If per chance that the context is not relevant you can remove the sources and just say 'internet' or 'general knowledge'")

        return llm.with_structured_output(AIResponseModel).invoke(prompt_template)


    def embed_list_of_text(self, text_list: list[str]) -> Union[list[list[float]], List[Tensor], ndarray, Tensor]:
        """Union[List[Tensor], ndarray, Tensor]
        This function embeds a list of text using a pre-trained model.
        :param text_list: The list of text to be embedded
        :return embeddings: The embeddings of the text
        """

        embeddings = self.embedding_model.encode(text_list, batch_size=16, convert_to_tensor=False)

        return embeddings

    def embed_text(self, text:str) -> Union[List[Tensor], ndarray, Tensor, None]:

        if not text.strip():
            return None

        return self.embedding_model.encode(text, batch_size=16, convert_to_tensor=False)

    def extract_context(self, data:dict):
        """
        Extracts and formats context from the given query response.
        Includes page number, source document, and content for each document chunk.

        :param data: A dictionary containing the query result.
        :return: A formatted string containing the extracted context.
        """
        context = ""

        # Extract the documents and metadata
        documents = data.get("documents", [[]])[0]  # Get the list of document chunks
        metadatas = data.get("metadatas", [[]])[0]  # Get the list of metadata chunks

        # Ensure the number of documents and metadatas match
        if len(documents) != len(metadatas):
            raise ValueError("Mismatch between the number of documents and metadata entries.")

        # Iterate over each document and its corresponding metadata
        for doc, metadata in zip(documents, metadatas):
            # Extract required metadata fields
            page_number = metadata.get("page_number", "Unknown Page")
            source_doc = metadata.get("from_pdf", "Unknown Document")

            # Append to the context string
            context += f"Source: {source_doc}, Page: {page_number}\n"
            context += f"Content: {doc}\n\n"

        return context

