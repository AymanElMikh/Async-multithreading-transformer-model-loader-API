🚀 Async Transformer Framework
A high-performance, production-ready framework for serving any Transformer model with async processing, intelligent batching, caching, and multi-model support.
✨ Key Features

🔄 Truly Async: Built on FastAPI with async/await for high concurrency
🎯 Universal Model Support: Works with any Transformer model (classification, generation, NER, Q&A, etc.)
📦 Intelligent Batching: Automatically batches requests for optimal GPU utilization
⚡ Response Caching: Built-in cache with configurable TTL
🔌 Dynamic Model Loading: Load/unload models on-the-fly without restart
📊 Multi-Model Support: Run multiple models simultaneously
🛡️ Production Ready: Error handling, health checks, statistics
🐳 Docker Support: Easy deployment with Docker & docker-compose

┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP Request
       ▼
┌─────────────────────────────────┐
│        FastAPI Layer            │
│  (Async request handling)       │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│     Inference Queue             │
│  • Request batching             │
│  • Cache management             │
│  • Async coordination           │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│     Model Manager               │
│  • Dynamic model loading        │
│  • Multi-model support          │
│  • Resource management          │
└─────────────────────────────────┘

Installation
# Clone the repository
git clone <your-repo>
cd transformer-framework

# Install dependencies
pip install -r requirements.txt

# Prepare your models
mkdir -p models/sentiment_model
# Copy your model files to models/sentiment_model/

Running the Server
# Start the server
python main.py

# Or with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

First API Call

# Check health
curl http://localhost:8000/

# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!", "model_id": "sentiment"}'
