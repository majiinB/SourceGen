from typing import Union, List
from dotenv import load_dotenv
import torch.cuda
from langchain_huggingface.llms import HuggingFacePipeline
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from torch import Tensor
from torch._numpy import ndarray
from transformers.utils import is_flash_attn_2_available

from app.models.response.ai_response import AIResponseModel

load_dotenv()

class AIService:
    """
    A service class for interacting with AI models, including embedding text and generating responses.

    Attributes:
        tokenizer (AutoTokenizer): The tokenizer for the gemma model.
        model_path (str): The path to the Hugging Face model for gemma.
        embedding_model (SentenceTransformer): The model used for embedding text.
    """

    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        """
        Initializes the AIService with the specified model.

        Args:
            model_name (str): The name of the default model to use for embedding text.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model_path = os.getenv("HUGGINGFACE_MODEL_GEMMA_PATH")
        self.embedding_model = SentenceTransformer(model_name_or_path=model_name, device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.local_gemma_model = self.__load_local_gemma_model()
        self.gemini_model = self.__load_gemini_flash_model()

    """   LOADING MODELS   """
    def __load_gemini_flash_model(self):
        """
            Loads the Gemini Flash model.

            :return: An instance of ChatGoogleGenerativeAI if successful, otherwise None.
        """
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.8,
                timeout=None,
                max_retries=2,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE
                }
            )
            return llm
        except Exception as e:
            print(f"ERROR: An error occurred while generating the model(online-gemini): {e}")
            return None

    def __load_local_gemma_model(self):
        """
            Loads the local Gemma model with appropriate attention implementation based on hardware capabilities.

            :return: An instance of HuggingFacePipeline if successful, otherwise None.
        """

        if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8):
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "sdpa"

        try:
            llm_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=self.model_path,
                                                             torch_dtype=torch.float16,
                                                             quantization_config=None,
                                                             low_cpu_mem_usage=False,  # use as much memory as we can
                                                             attn_implementation=attn_implementation).to("cuda")
            pipe = pipeline(
                task="text-generation",
                model=llm_model,
                device=0,
                tokenizer=self.tokenizer,
                max_new_tokens=500,
                framework="pt",
                model_kwargs={"temperature": 0.8, "do_sample": True}
            )
            return HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            print(f"ERROR: An error occurred while loading the model(local-gemma): {e}")
            return None

    """   PROMPT/QUERYING MODELS   """
    def prompt_gemini_flash(self, query: str, context: str) -> Union[AIResponseModel, None, dict]:
        """
        Generates a response using the Gemini Flash model based on the provided query and context.

        :param query: The user's query string.
        :param context: The context string to be used for generating the response.
        :return: An AIResponseModel containing the structured response or None if an error occurs.
        """
        try:
            llm = self.gemini_model
            return llm.with_structured_output(AIResponseModel).invoke(self.prompt_template_gen(query=query, context=context))
        except Exception as e:
            print(f"ERROR: An error occurred while generating the response: {e}")
            return None

    def prompt_gemini_flash_custom_response(self, query: str, context: str, response_model: Union[BaseModel, dict]) -> Union[AIResponseModel, None, dict]:
        """
        Generates a response using the Gemini Flash model based on the provided query and context.

        :param response_model:
        :param query: The user's query string.
        :param context: The context string to be used for generating the response.
        :return: An AIResponseModel containing the structured response or None if an error occurs.
        """
        try:
            llm = self.__load_gemini_flash()
            return llm.with_structured_output(response_model).invoke(self.prompt_template_gen(query=query, context=context))
        except Exception as e:
            print(f"ERROR: An error occurred while generating the response: {e}")
            return None

    def prompt_gemma(self, prompt: str, context: str) -> str|None:
        """
        Generates a response using the Gemini model based on the provided query and context.

        :param prompt: The user's query string.
        :param context: The context string to be used for generating the response.
        :return: An AIResponseModel containing the structured response or None if an error occurs.
        """
        try:
            return self.local_gemma_model.invoke(prompt)
        except Exception as e:
            print(f"ERROR: An error occurred while generating the response: {e}")
            return None

    """   EMBEDDING TEXT   """
    def embed_list_of_text(self, text_list: list[str]) -> Union[list[list[float]], List[Tensor], ndarray, Tensor]:
        """Union[List[Tensor], ndarray, Tensor]
        This function embeds a list of text using a pre-trained model.
        :param text_list: The list of text to be embedded
        :return embeddings: The embeddings of the text
        """
        embeddings = self.embedding_model.encode(text_list, batch_size=16, convert_to_tensor=False)
        return embeddings

    def embed_text(self, text: str) -> Union[List[Tensor], ndarray, Tensor, None]:
        """
        Embeds a single text string using a pre-trained model.

        :param text: The text to be embedded.
        :return: The embedding of the text as a list of tensors, ndarray, or tensor, or None if the text is empty.
        """
        if not text.strip():
            return None
        return self.embedding_model.encode(text, batch_size=16, convert_to_tensor=False)

    """ UTIL """
    def count_gemma_token(self, text:str) -> int:
        """
            Counts the number of tokens in the given text using the gemma tokenizer.
            :param text:str The text to be tokenized
            :return An int which is the number of tokens in the text.
        """
        input_ids = self.tokenizer(text, return_tensors="pt")
        return len(input_ids["input_ids"][0])

    """   EXTRACTING CONTEXT   """
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

    """   PROMPT TEMPLATES  """
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


