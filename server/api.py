from fastapi import FastAPI
from . import setup_logging
from .scheduler import RWKV070ModelLoader, DynamicScheduler

setup_logging()

app = FastAPI()

app