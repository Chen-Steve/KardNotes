from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, SecretStr
from typing import List
import openai
from nltk.tokenize import sent_tokenize
import os
import nltk
import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi import File, UploadFile
import PyPDF2
import io
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('punkt')

# Load environment variables at startup
load_dotenv()

# Get default API key from environment
DEFAULT_API_KEY = os.getenv('OPENAI_API_KEY')

app = FastAPI()

# Add CORS support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["kard.space"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BookContent(BaseModel):
    title: str
    content: str
    chapter: str = None

class Note(BaseModel):
    key_points: List[str]
    summary: str
    important_quotes: List[str]

class Config(BaseModel):
    openai_api_key: SecretStr

note_taker = None

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

# Add new endpoint for PDF processing
@app.post("/process-pdf", response_model=Note)
async def process_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # Read PDF content
        pdf_content = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        
        # Extract text from all pages
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Generate notes using existing functionality
        if not note_taker:
            raise HTTPException(status_code=400, detail="Service not configured")
        
        key_points = note_taker.extract_key_points(text)
        summary = note_taker.generate_summary(text)
        quotes = note_taker.extract_quotes(text)
        
        return Note(
            key_points=key_points,
            summary=summary,
            important_quotes=quotes
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))