from pydantic import BaseModel

class PageMetaData(BaseModel):
    """
            A class for storing page metadata

            Attributes:
            - from_pdf: str
            - page_number: int
            - page_char_count: int
            - page_word_count: int
            - page_sentence_count: int
            - page_token_count: float
    """
    from_pdf: str
    page_number: int
    page_char_count: int
    page_word_count: int
    page_sentence_count: int
    page_raw_token_count: int
    page_gemma_token_count: int
