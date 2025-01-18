from pydantic import BaseModel, Field


class AIResponseModel(BaseModel):
    question:str = Field(description="The question that i asked")
    answer:str = Field(description="The answer to the question or your response")
    references:str = Field(description="The references used, may be links or page number from the context given")
