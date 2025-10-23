"""
Example client for testing the Async Transformer Framework
Demonstrates single requests, batch requests, and concurrent usage
"""

import asyncio
import aiohttp
import time
from typing import List, Dict, Any


class TransformerClient:
    """Async client for the Transformer Framework API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the API is running"""
        async with self.session.get(f"{self.base_url}/") as response:
            return await response.json()
    
    async def predict(
        self,
        text: str,
        model_id: str = "sentiment",
        options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Single prediction request"""
        payload = {
            "text": text,
            "model_id": model_id,
            "options": options or {}
        }
        
        async with self.session.post(
            f"{self.base_url}/predict",
            json=payload
        ) as response:
            return await response.json()
    
    async def predict_batch(
        self,
        requests: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Batch prediction request"""
        async with self.session.post(
            f"{self.base_url}/predict/batch",
            json=requests
        ) as response:
            return await response.json()
    
    async def load_model(
        self,
        model_id: str,
        model_path: str,
        task_type: str,
        device: str = "auto"
    ) -> Dict[str, Any]:
        """Load a new model"""
        payload = {
            "model_id": model_id,
            "model_path": model_path,
            "task_type": task_type,
            "device": device
        }
        
        async with self.session.post(
            f"{self.base_url}/models/load",
            json=payload
        ) as response:
            return await response.json()
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all loaded models"""
        async with self.session.get(f"{self.base_url}/models") as response:
            return await response.json()
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        async with self.session.get(f"{self.base_url}/queue/stats") as response:
            return await response.json()
    
    async def clear_cache(self) -> Dict[str, Any]:
        """Clear the cache"""
        async with self.session.post(f"{self.base_url}/cache/clear") as response:
            return await response.json()


# Example Usage Functions

async def example_single_requests():
    """Example: Single prediction requests"""
    print("\n=== Example 1: Single Requests ===")
    
    async with TransformerClient() as client:
        # Check health
        health = await client.health_check()
        print(f"API Status: {health['status']}")
        print(f"Loaded models: {health['loaded_models']}")
        
        # Make predictions
        texts = [
            "I absolutely love this product!",
            "This is terrible, worst purchase ever.",
            "It's okay, nothing special."
        ]
        
        for text in texts:
            result = await client.predict(text, model_id="sentiment")
            print(f"\nText: {text}")
            print(f"Result: {result['result']}")
            print(f"Cached: {result['cached']}")
            print(f"Processing time: {result['processing_time']:.4f}s")


async def example_batch_request():
    """Example: Batch prediction"""
    print("\n=== Example 2: Batch Request ===")
    
    async with TransformerClient() as client:
        batch_texts = [
            {"text": "Great service!", "model_id": "sentiment"},
            {"text": "Not satisfied with quality", "model_id": "sentiment"},
            {"text": "Excellent experience", "model_id": "sentiment"},
            {"text": "Could be better", "model_id": "sentiment"},
        ]
        
        start = time.time()
        results = await client.predict_batch(batch_texts)
        elapsed = time.time() - start
        
        print(f"Processed {len(results)} texts in {elapsed:.4f}s")
        print(f"Average per text: {elapsed/len(results):.4f}s")
        
        for req, result in zip(batch_texts, results):
            print(f"\n'{req['text']}' -> {result['result']['label']} "
                  f"({result['result']['score']:.4f})")


async def example_concurrent_requests():
    """Example: High concurrency test"""
    print("\n=== Example 3: Concurrent Requests (Load Test) ===")
    
    async with TransformerClient() as client:
        # Create 100 concurrent requests
        num_requests = 100
        texts = [
            f"Test message number {i}" for i in range(num_requests)
        ]
        
        print(f"Sending {num_requests} concurrent requests...")
        start = time.time()
        
        # Send all requests concurrently
        tasks = [
            client.predict(text, model_id="sentiment")
            for text in texts
        ]
        results = await asyncio.gather(*tasks)
        
        elapsed = time.time() - start
        
        print(f"\n✅ Completed {num_requests} requests in {elapsed:.2f}s")
        print(f"Throughput: {num_requests/elapsed:.2f} req/s")
        print(f"Average latency: {elapsed/num_requests:.4f}s per request")
        
        # Check cache hits
        cached_count = sum(1 for r in results if r.get('cached'))
        print(f"Cache hits: {cached_count}/{num_requests} "
              f"({cached_count/num_requests*100:.1f}%)")


async def example_model_management():
    """Example: Dynamic model loading"""
    print("\n=== Example 4: Model Management ===")
    
    async with TransformerClient() as client:
        # List current models
        models = await client.list_models()
        print("Currently loaded models:")
        for model in models:
            print(f"  - {model['model_id']} ({model['task_type']}) "
                  f"on {model['device']}")
        
        # Get statistics
        stats = await client.get_queue_stats()
        print(f"\nQueue Statistics:")
        print(f"  Total requests: {stats['total_requests']}")
        print(f"  Cache hit rate: {stats['cache_hit_rate']}")
        print(f"  Avg batch size: {stats['avg_batch_size']:.2f}")
        print(f"  Cache size: {stats['cache_size']} entries")


async def example_cache_behavior():
    """Example: Demonstrating cache effectiveness"""
    print("\n=== Example 5: Cache Behavior ===")
    
    async with TransformerClient() as client:
        text = "This is a test message for caching"
        
        # First request (cache miss)
        print("First request (cache miss):")
        result1 = await client.predict(text)
        print(f"  Processing time: {result1['processing_time']:.4f}s")
        print(f"  Cached: {result1['cached']}")
        
        # Second request (cache hit)
        print("\nSecond request (cache hit):")
        result2 = await client.predict(text)
        print(f"  Processing time: {result2['processing_time']:.4f}s")
        print(f"  Cached: {result2['cached']}")
        
        # Calculate speedup
        if result1['processing_time'] > 0:
            speedup = result1['processing_time'] / max(0.0001, result2['processing_time'])
            print(f"\n⚡ Cache speedup: {speedup:.1f}x faster")


async def stress_test():
    """Stress test with multiple concurrent clients"""
    print("\n=== Stress Test: Multiple Concurrent Clients ===")
    
    async def client_worker(client_id: int, num_requests: int):
        """Simulate a client making multiple requests"""
        async with TransformerClient() as client:
            results = []
            for i in range(num_requests):
                result = await client.predict(
                    f"Client {client_id} message {i}",
                    model_id="sentiment"
                )
                results.append(result)
            return results
    
    # Run 10 clients, each making 20 requests
    num_clients = 10
    requests_per_client = 20
    total_requests = num_clients * requests_per_client
    
    print(f"Running {num_clients} clients, {requests_per_client} requests each...")
    print(f"Total requests: {total_requests}")
    
    start = time.time()
    
    # Run all clients concurrently
    client_tasks = [
        client_worker(i, requests_per_client)
        for i in range(num_clients)
    ]
    all_results = await asyncio.gather(*client_tasks)
    
    elapsed = time.time() - start
    
    print(f"\n✅ Completed {total_requests} requests in {elapsed:.2f}s")
    print(f"Throughput: {total_requests/elapsed:.2f} req/s")
    print(f"Average latency: {elapsed/total_requests:.4f}s")
    
    # Analyze results
    flat_results = [r for client_results in all_results for r in client_results]
    cached = sum(1 for r in flat_results if r.get('cached'))
    print(f"Cache hits: {cached}/{total_requests} ({cached/total_requests*100:.1f}%)")


# Main execution
async def main():
    """Run all examples"""
    print("🚀 Transformer Framework Client Examples")
    print("=" * 50)
    
    try:
        # Check if API is running
        async with TransformerClient() as client:
            await client.health_check()
        
        # Run examples
        await example_single_requests()
        await example_batch_request()
        await example_cache_behavior()
        await example_concurrent_requests()
        await example_model_management()
        
        # Optional: Run stress test
        print("\n" + "=" * 50)
        run_stress = input("\nRun stress test? (y/n): ").lower().strip()
        if run_stress == 'y':
            await stress_test()
        
        print("\n✅ All examples completed!")
        
    except aiohttp.ClientConnectorError:
        print("\n❌ Error: Cannot connect to API")
        print("Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())