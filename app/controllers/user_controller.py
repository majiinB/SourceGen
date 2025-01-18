from fastapi import APIRouter

router = APIRouter()

@router.get("/user")
def read_user():
    return {"username": "fakeuser"}