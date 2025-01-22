from pydantic import BaseModel

from app.models.document.page_metadata import PageMetaData

class PageContentModel(BaseModel):
    """
        A class for storing page contents

        Attributes:
        - id: str
        - page_metadata: PageMetaData
        - text: list[str]
        - embedding: list[list[float]]
    """
    id: str
    page_metadata: PageMetaData
    text: list[str]
    embedding: list[list[float]]