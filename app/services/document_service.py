import os.path
import time
import warnings
from langchain_community.document_loaders import PyMuPDFLoader
from spacy.lang.en import English
from dotenv import load_dotenv
import os
from app.models.document.page_content import PageContentModel
from app.models.document.page_metadata import PageMetaData
from app.models.document.pdf_document_content import PdfDocumentContentModel
from app.models.response.response import ResponseModel
from app.repositories.document_repository import DocumentRepository
from app.services.ai_service import AIService
from app.utils.document_utils import multiple_text_formater, split_text_with_separators, text_formater

load_dotenv()

class DocumentService:
    """
        A service class for loading and processing PDF documents.
    """

    def __init__(self):
        self.ai_service = AIService()

    async def load_and_process_pdf_document(self, pdf_path: str, collection_name:str, start_page:int) -> ResponseModel:
        """
            Loads and processes a PDF document.
            :param start_page:
            :param collection_name:
            :param pdf_path: The path to the PDF document.
            :return PdfDocumentContentModel: A model containing the document title, number of pages in document, and page content.
        """

        # Initialize the list that will hold the pages and their respective text
        pages_and_texts = []

        # Check if the file exists
        if not os.path.exists(pdf_path):
            return None

        # Initialize the spacy model
        nlp = English()
        nlp.add_pipe("sentencizer")

        loader = PyMuPDFLoader(pdf_path) # Load the pdf file

        page_number = 1 # Initialize the page number
        current_loader_page = 0  # Tracks the loader's current page (independent of page_number)
        page_content_holder: list[PageContentModel]=[]  # Initialize the page content holder
        skipped_empty_page: list[int] = []
        skipped_low_token_count_page: list[str] = []

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*Empty content on page.*"
            )

            # Load and iterate through the pages
            iterator_start = time.time()  # Start timer for iterator
            async for page in loader.alazy_load():

                # Skip pages before the start_page
                if current_loader_page < start_page:
                    skipped_empty_page.append(current_loader_page)  # Optional: Log the skipped page
                    current_loader_page += 1  # Increment the loader page number
                    continue

                if not page.page_content.strip():  # Check if the page content is empty or whitespace
                    skipped_empty_page.append(page_number)  # Optional: Log the skipped page
                    page_number+=1
                    continue  # Skip this iteration and move to the next page

                cleaned_text = text_formater(page.page_content) # Clean the text for stats

                try:
                    sentence_item = list(nlp(cleaned_text).sents)  # Split the text into sentences
                    sentence_item = [str(sentence) for sentence in sentence_item]  # Make sure the sentences are strings
                except Exception as e:
                    print(f"SPACY ERROR: {e}")
                    return ResponseModel(
                        status=500,
                        message=f"An error has occurred while processing(counting sentences) page {page_number}",
                        data=None
                    )

                page_text = split_text_with_separators(page.page_content) # Split the text into chunks
                page_text: list[str] = multiple_text_formater(page_text) # Clean the text for storing

                # Count token for to determine if page is relevant
                raw_token_count = round(len(cleaned_text.replace(" ", "")) / 4)
                try:
                    gemma_token_count = self.ai_service.count_gemma_token(cleaned_text)
                except Exception as e:
                    print(f"GEMMA TOKEN COUNTER ERROR: {e}")
                    return ResponseModel(
                        status=500,
                        message=f"An error has occurred while processing(counting tokens using gemma) page {page_number}",
                        data=None
                    )

                if raw_token_count < 30 and gemma_token_count < 30 :
                    skipped_low_token_count_page.append({f"page_number: {page_number} raw_token_count: {raw_token_count} "
                                                         f"gemma_token_count: {gemma_token_count} content: {cleaned_text}"})
                    page_number+=1
                    continue

                page_metadata = PageMetaData(
                    from_pdf = page.metadata["source"].replace("data/","").replace(".pdf",""),
                    page_number=page_number,
                    page_char_count=len(cleaned_text),
                    page_word_count=len(cleaned_text.split()),
                    page_sentence_count=len(sentence_item),  # Count the number of sentences using the spacy model
                    page_raw_token_count=raw_token_count,
                    page_gemma_token_count=gemma_token_count
                )

                # Append the page to the list
                embed_sentence_start = time.time()
                try:
                    page_content = PageContentModel(
                        id=f"page_{page_number}",
                        page_metadata=page_metadata,
                        text=page_text,
                        embedding=self.ai_service.embed_list_of_text(text_list=page_text)  # Embed page text
                    )
                    print(f"Embedding text of page {page_number} took: {time.time() - embed_sentence_start:.2f} seconds\n")
                except Exception as e:
                    time.time()
                    print(f"EMBEDDING ERROR: {e}")
                    return ResponseModel(
                        status=500,
                        message=f"An error has occurred while processing(embedding) page {page_number}",
                        data=None
                    )

                page_content_holder.append(page_content)
                page_number += 1 # Increment the page number

        # End Timer for iterator
        print(f"Iterating through the PDF took: {time.time() - iterator_start:.2f} seconds\n")

        # For Debug and additional info
        print(f"Skipped Empty Pages: {skipped_empty_page}")
        print(f"Skipped low token count {skipped_low_token_count_page}")

        if not page_content_holder:
            return ResponseModel(
                status=400,
                message="PDF Content is empty",
                data={
                    "skipped_empty_pages": skipped_empty_page,
                    "skipped_low_token_count": skipped_low_token_count_page
                }
            )

        pdf_document_content = PdfDocumentContentModel(
            title=pdf_path.replace(os.getenv("UPLOAD_FILE_PATH"),"").replace(".pdf",""),
            number_of_pages=len(page_content_holder),
            page_content=page_content_holder,
        )

        # Store to vector database and then return a response object
        try:
            response = (DocumentRepository(collection_name=collection_name)
                        .store_pdf_document_in_vector_chroma(pdf_document_content=pdf_document_content))
        except Exception as e:
            print(f"STORAGE ERROR: {e}")
            return ResponseModel(
                status=500,
                message=f"An error has occurred while storing the contents of page {page_number}",
                data=None
            )

        response.data = {
            "skipped_empty_pages": skipped_empty_page,
            "skipped_low_token_count": skipped_low_token_count_page
        }
        return response

    def query_document(self, query: str, collection_name: str) -> ResponseModel:
        embedded_query = self.ai_service.embed_text(query)

        if not embedded_query is None:
            data = DocumentRepository(collection_name=collection_name).query_vector_chroma_db(embedded_query=embedded_query)
            context = self.ai_service.extract_context(data)
            data = self.ai_service.prompt_gemini_flash(query=query, context=context)
            return ResponseModel(
                status=200,
                message="Query successful",
                data=data
            )
        else:
            return ResponseModel(
                status=400,
                message="Empty query",
                data=None
            )

