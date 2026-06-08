from fastapi import APIRouter
from .rwkv import router as rwkv_router
from .openai import router as openai_router

router = APIRouter(prefix="/v1")

router.include_router(rwkv_router)
router.include_router(openai_router)