from typing import List
import re
from openai import OpenAI
import tiktoken

class OpenAINoteTaker:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        
    def estimate_token_count(self, text: str) -> int:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
        
    def chunk_text(self, text: str, max_tokens: int = 2000) -> List[str]:
        """Split text into chunks based on token limits."""
        # Simple splitting by periods, exclamation marks, and question marks
        sentences = re.split('[.!?]', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:  # Skip empty sentences
                continue
                
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
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts key points from text."},
                    {"role": "user", "content": f"Extract 5 key points from this text: {text}"}
                ],
                temperature=0.3,
                max_tokens=300
            )
            key_points_text = response.choices[0].message.content.strip()
            key_points = re.split(r"\n|[0-9]+\.", key_points_text)
            return [point.strip() for point in key_points if point.strip()]
        except Exception as e:
            print("Error extracting key points:", e)
            return []

    def generate_summary(self, text: str) -> str:
        chunks = self.chunk_text(text)
        summaries = []
        
        for chunk in chunks:
            try:
                response = self.client.chat.completions.create(
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
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that identifies important quotes."},
                    {"role": "user", "content": f"Extract 3 important quotes from this text that represent key ideas or concepts: {text}"}
                ],
                temperature=0.3,
                max_tokens=200
            )
            quotes_text = response.choices[0].message.content.strip()
            quotes = re.split(r"\n|[0-9]+\.", quotes_text)
            return [quote.strip().strip('"\'') for quote in quotes if quote.strip()]
        except Exception as e:
            print("Error extracting quotes:", e)
            return []
