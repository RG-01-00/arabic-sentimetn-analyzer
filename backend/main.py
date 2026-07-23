from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from sentiment_engine import (
    analyze_arabic_sentiment,
    get_analyzer
)


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Loading Arabic BERT model...")

    get_analyzer()

    print("Arabic BERT model loaded successfully!")

    yield


app = FastAPI(
    title="Arabic Sentiment Analysis API",
    description="Arabic NLP sentiment analysis engine",
    version="1.0.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):

    text: str


@app.get("/")
def root():

    return {
        "status": "online",
        "message": "Arabic Sentiment Analysis API is running",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "message": "Arabic Sentiment Analysis API is running",
    }


@app.get("/ready")
def ready():

    get_analyzer()

    return {
        "status": "ready",
        "message": "Arabic BERT model is loaded"
    }


@app.post("/analyze")
def analyze(request: AnalyzeRequest):

    result = analyze_arabic_sentiment(request.text)

    return result