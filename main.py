"""
FastAPI-based Async Transformer Model Framework
Supports multiple models, batching, and high concurrency
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import asyncio
from contextlib import asynccontextmanager

from model_manager import ModelManager
from inference import InferenceQueue

# Global instances
model_manager: ModelManager = None
inference_queue: InferenceQueue = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global model_manager, inference_queue
    
    # Startup
    print("🚀 Starting Transformer Framework...")
    model_manager = ModelManager()
    inference_queue = InferenceQueue(model_manager, batch_size=8, wait_time=0.1)
    
    # Load default models from config
    await model_manager.load_model(
        model_id="sentiment",
        model_path="./models/sentiment_model",
        task_type="text-classification"
    )
    
    # Start the inference queue processor
    asyncio.create_task(inference_queue.process_queue())
    
    print("✅ Framework ready!")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down...")
    await inference_queue.stop()
    model_manager.cleanup()


app = FastAPI(
    title="Async Transformer Framework",
    description="High-performance API for transformer model inference",
    version="1.0.0",
    lifespan=lifespan
)


# Request/Response Models
class InferenceRequest(BaseModel):
    text: str = Field(..., description="Input text for inference")
    model_id: str = Field(default="sentiment", description="Model identifier")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Additional inference options")


class InferenceResponse(BaseModel):
    model_id: str
    result: Dict[str, Any]
    cached: bool = False
    processing_time: float


class ModelLoadRequest(BaseModel):
    model_id: str = Field(..., description="Unique identifier for the model")
    model_path: str = Field(..., description="Path to model directory")
    task_type: str = Field(..., description="Type: text-classification, text-generation, ner, qa, etc.")
    device: Optional[str] = Field(default="auto", description="Device: cpu, cuda, auto")


class ModelInfo(BaseModel):
    model_id: str
    task_type: str
    loaded: bool
    device: str
    memory_usage_mb: Optional[float]


# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "framework": "Async Transformer Framework",
        "loaded_models": list(model_manager.models.keys()) if model_manager else []
    }


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """
    Main inference endpoint with automatic batching and caching
    """
    if not model_manager.is_model_loaded(request.model_id):
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model_id}' not found. Load it first via /models/load"
        )
    
    try:
        # Submit to queue for batched processing
        result = await inference_queue.submit_request(
            model_id=request.model_id,
            text=request.text,
            options=request.options or {}
        )
        
        return InferenceResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/predict/batch", response_model=List[InferenceResponse])
async def predict_batch(requests: List[InferenceRequest]):
    """
    Batch prediction endpoint for multiple requests at once
    """
    tasks = []
    for req in requests:
        if not model_manager.is_model_loaded(req.model_id):
            raise HTTPException(
                status_code=404,
                detail=f"Model '{req.model_id}' not found"
            )
        tasks.append(
            inference_queue.submit_request(
                model_id=req.model_id,
                text=req.text,
                options=req.options or {}
            )
        )
    
    try:
        results = await asyncio.gather(*tasks)
        return [InferenceResponse(**r) for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")


@app.post("/models/load")
async def load_model(request: ModelLoadRequest, background_tasks: BackgroundTasks):
    """
    Dynamically load a new model
    """
    if model_manager.is_model_loaded(request.model_id):
        return {"message": f"Model '{request.model_id}' already loaded"}
    
    try:
        await model_manager.load_model(
            model_id=request.model_id,
            model_path=request.model_path,
            task_type=request.task_type,
            device=request.device
        )
        return {
            "message": f"Model '{request.model_id}' loaded successfully",
            "model_id": request.model_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.delete("/models/{model_id}")
async def unload_model(model_id: str):
    """
    Unload a model to free resources
    """
    if not model_manager.is_model_loaded(model_id):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    
    try:
        model_manager.unload_model(model_id)
        return {"message": f"Model '{model_id}' unloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """
    List all loaded models and their info
    """
    return model_manager.get_models_info()


@app.get("/models/{model_id}", response_model=ModelInfo)
async def get_model_info(model_id: str):
    """
    Get detailed info about a specific model
    """
    if not model_manager.is_model_loaded(model_id):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    
    return model_manager.get_model_info(model_id)


@app.get("/queue/stats")
async def queue_stats():
    """
    Get current queue statistics
    """
    return inference_queue.get_stats()


@app.post("/cache/clear")
async def clear_cache():
    """
    Clear the inference cache
    """
    cleared = inference_queue.clear_cache()
    return {"message": f"Cache cleared", "entries_removed": cleared}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Use 1 worker to share model memory
        reload=False
    )