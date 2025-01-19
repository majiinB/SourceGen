from typing import Union, List
from dotenv import load_dotenv
import torch.cuda
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
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
        self.model_path=os.getenv("HUGGINGFACE_MODEL_GEMMA_PATH")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(model_name_or_path=model_name, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)


    def prompt_gemini_flash(self, query:str, context:str):
        llm = ChatGoogleGenerativeAI(
            model = "gemini-1.5-flash",
            temperature=0,
            timeout=None,
            max_retries=2,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE
            }
        )

        return llm.with_structured_output(AIResponseModel).invoke(self.prompt_template_gen(query=query, context=context))


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

    def count_gemma_token(self, text:str) -> int:
        input_ids = self.tokenizer(text, return_tensors="pt")
        return len(input_ids["input_ids"][0])

    def prompt_template_gen(self, query:str, context:str) -> str:
        prompt = (f"Using the provided context, answer the user's query in the structured format described below.\n"
                  f"Follow these steps to ensure accuracy and clarity:\n"
                  f"1. Identify and extract the most relevant passages from the context to answer the question.\n"
                  f"2. Construct a detailed and explanatory answer based on the extracted context.\n"
                  f"3. Clearly cite the references used in your answer, such as page numbers or links from the context.\n"
                  f"4. If the provided context is irrelevant to the question, base your answer on general knowledge and indicate the reference as 'general knowledge' or 'internet'.\n\n"
                  f"Here is the context you may use to answer the query:\n\n"
                  f"Context:\n{context}\n\n"
                  f"Question: {query}\n\n"
                  f"Provide your response strictly in the following structured format:\n\n"
                  f"### Structured Output Format:\n"
                  f"{{\n"
                  f"  \"question\": \"<Insert the question provided>\",\n"
                  f"  \"answer\": \"<Insert your detailed answer here>\",\n"
                  f"  \"references\": \"<Insert your references here, such as page numbers, links, or 'general knowledge/internet'>\"\n"
                  f"}}\n\n"
                  f"### Example 1 (Using Context):\n"
                  f"{{\n"
                  f"  \"question\": \"What are macronutrients?\",\n"
                  f"  \"answer\": \"Macronutrients are nutrients required in large amounts by the body for energy and growth. They include proteins, fats, and carbohydrates.\",\n"
                  f"  \"references\": \"page 12\"\n"
                  f"}}\n\n"
                  f"### Example 2 (General Knowledge):\n"
                  f"{{\n"
                  f"  \"question\": \"What are macronutrients?\",\n"
                  f"  \"answer\": \"Macronutrients are nutrients required in large amounts by the body for energy and growth. They include proteins, fats, and carbohydrates.\",\n"
                  f"  \"references\": \"general knowledge/internet\"\n"
                  f"}}\n\n"
                  f"Now, based on the provided context and question, provide your response in the structured format:")

        return prompt

