from fastapi import APIRouter, File, UploadFile, Form
from dotenv import load_dotenv
from app.models.response import ResponseModel
from app.services.document_service import DocumentService
from app.utils.document_utils import is_valid_collection_name
import os

router = APIRouter()
load_dotenv()

@router.post("/v1/user/upload_pdf")
async def upload_or_load_doc(file: UploadFile = File(...), start_page: int = Form(...)):
    """
    Endpoint to upload and process a PDF document.

    This endpoint allows users to upload a PDF document, which is then saved to the server and processed.
    The processing includes loading the PDF, extracting text, and storing the content in a vector database.

    Args:
        file (UploadFile): The PDF file to be uploaded and processed.

    Returns:
        ResponseModel: A response model containing the status, message, and data of the operation.
        :param file:
        :param start_page:
    """
    collection_name = file.filename.replace(".pdf","")

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

    file_path = f"{os.getenv("UPLOAD_FILE_PATH")}/{collection_name}.pdf"

    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

    return await DocumentService().load_and_process_pdf_document(
        pdf_path=file_path,
        collection_name=collection_name, start_page=(start_page-1))
