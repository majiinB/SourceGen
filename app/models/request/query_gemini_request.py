from pydantic import BaseModel, Field, field_validator 

class QueryGeminiRequestModel(BaseModel):
    query: str = Field(..., min_length=1, description="The query string to search for")
    collection_name: str = Field(..., min_length=1, description="The name of the collection")

    @field_validator("query", "collection_name")
    def non_empty(cls, value):
        if not value.strip():  # Ensure the string is not just whitespace
            raise ValueError("Field cannot be empty or whitespace")
        return value
