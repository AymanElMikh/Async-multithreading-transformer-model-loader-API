"""
Inference Queue - Handles request batching, caching, and async processing
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timedelta


@dataclass
class InferenceRequest:
    """Single inference request"""
    model_id: str
    text: str
    options: Dict[str, Any]
    future: asyncio.Future
    submitted_at: float = field(default_factory=time.time)
    
    def cache_key(self) -> str:
        """Generate cache key for this request"""
        data = {
            "model_id": self.model_id,
            "text": self.text,
            "options": self.options
        }
        return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


class InferenceQueue:
    """
    Manages inference requests with intelligent batching and caching
    """
    
    def __init__(
        self,
        model_manager,
        batch_size: int = 8,
        wait_time: float = 0.1,
        cache_ttl: int = 3600
    ):
        """
        Args:
            model_manager: ModelManager instance
            batch_size: Maximum batch size for inference
            wait_time: Time to wait for batching (seconds)
            cache_ttl: Cache time-to-live (seconds)
        """
        self.model_manager = model_manager
        self.batch_size = batch_size
        self.wait_time = wait_time
        self.cache_ttl = cache_ttl
        
        # Queue for each model
        self.queues: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        
        # Cache: {cache_key: (result, timestamp)}
        self.cache: Dict[str, tuple[Dict[str, Any], float]] = {}
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "batches_processed": 0,
            "avg_batch_size": 0.0
        }
        
        self.running = True
    
    async def submit_request(
        self,
        model_id: str,
        text: str,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Submit an inference request (returns when completed)
        """
        self.stats["total_requests"] += 1
        
        # Create request
        future = asyncio.Future()
        request = InferenceRequest(
            model_id=model_id,
            text=text,
            options=options,
            future=future
        )
        
        # Check cache
        cache_key = request.cache_key()
        cached_result = self._get_from_cache(cache_key)
        
        if cached_result is not None:
            self.stats["cache_hits"] += 1
            return {
                "model_id": model_id,
                "result": cached_result,
                "cached": True,
                "processing_time": 0.0
            }
        
        self.stats["cache_misses"] += 1
        
        # Add to queue
        await self.queues[model_id].put(request)
        
        # Wait for result
        result = await future
        return result
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get result from cache if valid"""
        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return result
            else:
                # Expired
                del self.cache[cache_key]
        return None
    
    def _add_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """Add result to cache"""
        self.cache[cache_key] = (result, time.time())
    
    async def process_queue(self):
        """Main queue processing loop"""
        print("🔄 Starting inference queue processor...")
        
        while self.running:
            # Process each model's queue
            tasks = []
            for model_id in list(self.queues.keys()):
                if not self.queues[model_id].empty():
                    tasks.append(self._process_model_queue(model_id))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            else:
                await asyncio.sleep(0.01)  # Small sleep to prevent busy waiting
    
    async def _process_model_queue(self, model_id: str):
        """Process pending requests for a specific model"""
        queue = self.queues[model_id]
        
        # Collect batch
        batch: List[InferenceRequest] = []
        deadline = time.time() + self.wait_time
        
        while len(batch) < self.batch_size and time.time() < deadline:
            try:
                request = await asyncio.wait_for(
                    queue.get(),
                    timeout=max(0.01, deadline - time.time())
                )
                batch.append(request)
            except asyncio.TimeoutError:
                break
        
        if not batch:
            return
        
        # Process batch
        try:
            await self._process_batch(model_id, batch)
        except Exception as e:
            # Propagate error to all futures in batch
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)
    
    async def _process_batch(self, model_id: str, batch: List[InferenceRequest]):
        """Process a batch of requests"""
        start_time = time.time()
        
        # Get model
        model_instance = self.model_manager.get_model(model_id)
        
        # Prepare inputs
        texts = [req.text for req in batch]
        
        # Run inference in thread pool (blocking operation)
        results = await asyncio.to_thread(
            self._run_inference,
            model_instance,
            texts
        )
        
        processing_time = time.time() - start_time
        
        # Update stats
        self.stats["batches_processed"] += 1
        self.stats["avg_batch_size"] = (
            (self.stats["avg_batch_size"] * (self.stats["batches_processed"] - 1) + len(batch))
            / self.stats["batches_processed"]
        )
        
        # Return results to futures
        for request, result in zip(batch, results):
            response = {
                "model_id": model_id,
                "result": result,
                "cached": False,
                "processing_time": processing_time / len(batch)
            }
            
            # Cache result
            cache_key = request.cache_key()
            self._add_to_cache(cache_key, result)
            
            # Complete future
            if not request.future.done():
                request.future.set_result(response)
    
    def _run_inference(self, model_instance, texts: List[str]) -> List[Dict[str, Any]]:
        """Run actual inference (synchronous)"""
        # Use pipeline for inference
        results = model_instance.pipeline(texts)
        
        # Normalize results based on task type
        normalized = []
        for result in results:
            if isinstance(result, list):
                # For tasks like NER that return lists
                normalized.append({"entities": result})
            elif isinstance(result, dict):
                # For classification tasks
                if "label" in result and "score" in result:
                    normalized.append({
                        "label": result["label"],
                        "score": float(result["score"])
                    })
                else:
                    normalized.append(result)
            else:
                normalized.append({"output": str(result)})
        
        return normalized
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        queue_sizes = {
            model_id: queue.qsize()
            for model_id, queue in self.queues.items()
        }
        
        cache_hit_rate = (
            self.stats["cache_hits"] / max(1, self.stats["total_requests"])
        ) * 100
        
        return {
            **self.stats,
            "cache_hit_rate": f"{cache_hit_rate:.2f}%",
            "cache_size": len(self.cache),
            "queue_sizes": queue_sizes,
            "active_models": len([q for q in self.queues.values() if q.qsize() > 0])
        }
    
    def clear_cache(self) -> int:
        """Clear the cache and return number of entries removed"""
        count = len(self.cache)
        self.cache.clear()
        return count
    
    async def stop(self):
        """Stop the queue processor"""
        self.running = False
        print("⏹️  Inference queue stopped")