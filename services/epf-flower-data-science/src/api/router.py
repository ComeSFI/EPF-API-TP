"""API Router for Fast API."""
from fastapi import APIRouter
from fastapi.responses import RedirectResponse

from src.api.routes import hello
from src.api.routes import data

router = APIRouter()

@router.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")



router.include_router(hello.router, tags=["Hello"])

router.include_router(data.router, tags=["Download Data"])
