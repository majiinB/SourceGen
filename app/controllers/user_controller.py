from fastapi import APIRouter, File, UploadFile, Form, Body
from dotenv import load_dotenv

from app.models.query_gemini_request import QueryGeminiRequestModel
from app.models.response import ResponseModel
from app.services.document_service import DocumentService
from app.utils.document_utils import is_valid_collection_name
import os

router = APIRouter()
load_dotenv()
document_service = DocumentService()

@router.post("/v1/user/upload_pdf")
async def upload_or_load_doc(file: UploadFile = File(...), start_page: int = Form(...)):
    """
    Endpoint to upload and process a PDF document.

    This endpoint allows users to upload a PDF document, which is then saved to the server and processed.
    The processing includes loading the PDF, extracting text, and storing the content in a vector database.

    Args:
        :param file: (UploadFile) The PDF file to be uploaded and processed.
        :param start_page: int The page where the actual content of the pdf start

    Returns:
        ResponseModel: A response model containing the status, message, and data of the operation.
    """

    # Collection name defaults to the name of the pdf file
    collection_name = file.filename.replace(".pdf","")

    # Shield
    if start_page < 1:
        return ResponseModel(
            status=400,
            message="start page can't be less than 1",
            data=None
        )

    if file.content_type != "application/pdf":
        return ResponseModel(
            status=400,
            message="Invalid document type",
            data=None
        )

    if not is_valid_collection_name(collection_name):
        return ResponseModel(
            status=400,
            message="Invalid PDF document name",
            data=None
        )

    # Initialize the path where the uploaded file will be stored
    file_path = f"{os.getenv("UPLOAD_FILE_PATH")}/{collection_name}.pdf"

    # Check if file already exists, if it exists proceed to processing the contents,
    # else read the uploaded file and save locally
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

    return await document_service.load_and_process_pdf_document(
        pdf_path=file_path,
        collection_name=collection_name, start_page=(start_page-1))

@router.post("/v1/user/query_gemini")
async def query_gemini(payload: QueryGeminiRequestModel = Body(...)):
    """
    Endpoint to query the AI (Gemini), as well as the vector database for context.

    This endpoint allows users to query the vector database for a document using a text query.
    The query is embedded into a vector and compared against the vectors in the database to find the most similar document.
    The context of the document is then extracted and used to generate a response using the Gemini AI model.

    Args:
        :param payload: QueryGeminiRequestModel The payload containing the query and collection name.

    Returns:
        ResponseModel: A response model containing the status, message, and data of the operation.
    """
    query = payload.query
    collection_name = payload.collection_name

    return document_service.query_document(query=query, collection_name=collection_name)
