from fastapi import FastAPI
from .controllers import user_controller
from .models.response import ResponseModel
from .repositories.document_repository import DocumentRepository
from .services.document_service import DocumentService
from .utils.document_utils import is_valid_collection_name

app = FastAPI()

@app.get("/ewanko")
async def read_root():
    collection_name = "human-nutrition-text"
    if not is_valid_collection_name(collection_name):
        return ResponseModel(
            status=400,
            message="Invalid collection name",
            data=None
        )
    # retrieval = await DocumentService().load_and_process_pdf_document(pdf_path="data/human-nutrition-text.pdf",
    #                                                                   collection_name=collection_name, start_page=43)
    # data = DocumentRepository(collection_name=collection_name).get_collection()
    # DocumentService().query_document(query="what are the best sources of protein", collection_name=collection_name)
    # print(data)
    return DocumentService().query_document(query="what are the best sources of protein", collection_name=collection_name)
    # return retrieval

app.include_router(user_controller.router)


