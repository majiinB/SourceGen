from pydantic import BaseModel, Field

class AIResponseModel(BaseModel):
    """
        A model representing the AI response.

        Attributes:
            question (str): The question that was asked.
            answer (str): The detailed answer to the question.
            references (str): The references used, such as links or page numbers from the provided context.
    """
    question: str = Field(description="The question that was asked.")
    answer: str = Field(description="The detailed answer to the question.")
    references: str = Field(description="The references used, such as links or page numbers from the provided context.")