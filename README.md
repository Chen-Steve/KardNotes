# KardNote

A lightweight microservice that generates human-like notes from book content using OpenAI's & Anthropic Embedding. This service is part of the Kard flashcard application ecosystem.

- Extracts key points from text
- Generates concise summaries
- Identifies important quotes
- RESTful API endpoints
- Docker containerized

### API Endpoints
- `POST /configure` - Set up OpenAI API key
- `POST /generate-notes` - Generate notes from book content
- `GET /health` - Service health check
- `GET /` - API information