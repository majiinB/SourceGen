from pydantic import BaseModel

class ResponseModel(BaseModel):
    status: int
    message: str
    data: object | None