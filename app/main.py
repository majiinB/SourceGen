from fastapi import FastAPI
from .controllers import user_controller
from .models.response.response import ResponseModel

app = FastAPI()

@app.get("/")
async def read_root():
    return ResponseModel(
        status=200,
        message="SourceGen is online",
        data=None
    )

app.include_router(user_controller.router)


