from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, SecretStr
from typing import List
import os
import logging
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.note_taker import OpenAINoteTaker

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables at startup
load_dotenv()

# Get default API key from environment and initialize note_taker
DEFAULT_API_KEY = os.getenv('OPENAI_API_KEY')
if DEFAULT_API_KEY:
    note_taker = OpenAINoteTaker(DEFAULT_API_KEY)
else:
    raise Exception("OPENAI_API_KEY environment variable is required")

app = FastAPI()

# Add CORS support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["kard.space", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add static files middleware
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Add route to serve index.html
@app.get("/ui")
async def serve_ui():
    return FileResponse("app/static/index.html")

class BookContent(BaseModel):
    content: str
    chapter: str = None

class Note(BaseModel):
    key_points: List[str]
    summary: str
    important_quotes: List[dict]

class Config(BaseModel):
    openai_api_key: SecretStr

@app.post("/configure")
async def configure(config: Config = None):
    global note_taker
    api_key = config.openai_api_key.get_secret_value() if config else DEFAULT_API_KEY
    
    if not api_key:
        raise HTTPException(
            status_code=400, 
            detail="No API key provided. Either configure with POST or set OPENAI_API_KEY environment variable."
        )
    
    note_taker = OpenAINoteTaker(api_key)
    return {"status": "configured"}

@app.post("/generate-notes", response_model=Note)
async def generate_notes(book_content: BookContent):
    if not note_taker:
        raise HTTPException(status_code=400, detail="Service not configured. Please call /configure first.")
    
    try:
        key_points = note_taker.extract_key_points(book_content.content)
        summary = note_taker.generate_summary(book_content.content)
        quotes = note_taker.extract_quotes(book_content.content)
        
        return Note(
            key_points=key_points,
            summary=summary,
            important_quotes=quotes
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint accessed")
    return {"status": "healthy"}

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {
        "message": "Welcome to the Note Taker API",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "configure": "/configure",
            "generate-notes": "/generate-notes"
        }
    }