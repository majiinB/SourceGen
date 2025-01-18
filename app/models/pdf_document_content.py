from datetime import datetime

from pandas.core.interchange.dataframe_protocol import DataFrame
from pydantic import BaseModel

from app.models.page_content import PageContentModel

class PdfDocumentContentModel(BaseModel):
    """
            A class for storing PDF contents

            Attributes:
            - title: str
            - number_of_pages: int
            - page_content: list[PageContentModel]
    """
    title: str
    number_of_pages: int
    page_content: list[PageContentModel]