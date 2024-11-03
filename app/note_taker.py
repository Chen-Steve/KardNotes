from typing import List
import openai
from nltk.tokenize import sent_tokenize
import tiktoken
import re

class OpenAINoteTaker:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        
    def estimate_token_count(self, text: str) -> int:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
        
    def chunk_text(self, text: str, max_tokens: int = 2000) -> List[str]:
        """Split text into chunks based on token limits to avoid truncation."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = self.estimate_token_count(sentence)
            if current_length + sentence_length > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def extract_key_points(self, text: str) -> List[str]:
        prompt = f"Extract 5 key points from the following text. Text: {text}"
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts key points from text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            key_points_text = response.choices[0].message.content.strip()
            key_points = re.split(r"\n|[0-9]+\.", key_points_text)  # Handles numbered points or line breaks
            return [point.strip() for point in key_points if point.strip()]
        except Exception as e:
            print("Error extracting key points:", e)
            return []

    def generate_summary(self, text: str) -> str:
        chunks = self.chunk_text(text)
        summaries = []
        
        for chunk in chunks:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                        {"role": "user", "content": f"Provide a brief summary of this text: {chunk}"}
                    ],
                    temperature=0.3,
                    max_tokens=150
                )
                summaries.append(response.choices[0].message.content.strip())
            except Exception as e:
                print("Error generating summary:", e)
        
        return " ".join(summaries)

    def extract_quotes(self, text: str) -> List[str]:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that identifies important quotes."},
                    {"role": "user", "content": f"Extract 3 important quotes from this text that represent key ideas or concepts: {text}"}
                ],
                temperature=0.3,
                max_tokens=200
            )
            quotes_text = response.choices[0].message.content.strip()
            quotes = re.split(r"\n|[0-9]+\.", quotes_text)  # Handle numbered points or line breaks
            return [quote.strip().strip('"\'') for quote in quotes if quote.strip()]
        except Exception as e:
            print("Error extracting quotes:", e)
            return []
