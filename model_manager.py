"""
Model Manager - Handles loading, unloading, and managing multiple transformer models
"""

import os
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    pipeline
)
from typing import Dict, Optional, Any, List
import asyncio
from dataclasses import dataclass
from datetime import datetime
import psutil


@dataclass
class ModelInstance:
    """Container for model information"""
    model_id: str
    model: Any
    tokenizer: Any
    pipeline: Any
    task_type: str
    device: str
    loaded_at: datetime
    last_used: datetime


class ModelManager:
    """Manages multiple transformer models with dynamic loading/unloading"""
    
    TASK_MODEL_MAPPING = {
        "text-classification": AutoModelForSequenceClassification,
        "sentiment-analysis": AutoModelForSequenceClassification,
        "text-generation": AutoModelForCausalLM,
        "ner": AutoModelForTokenClassification,
        "token-classification": AutoModelForTokenClassification,
        "question-answering": AutoModelForQuestionAnswering,
        "qa": AutoModelForQuestionAnswering,
    }
    
    def __init__(self):
        self.models: Dict[str, ModelInstance] = {}
        self.lock = asyncio.Lock()
    
    async def load_model(
        self,
        model_id: str,
        model_path: str,
        task_type: str,
        device: str = "auto"
    ) -> None:
        """
        Load a transformer model from local path
        
        Args:
            model_id: Unique identifier for this model instance
            model_path: Path to the model directory
            task_type: Type of task (text-classification, text-generation, etc.)
            device: Device to load on (cpu, cuda, auto)
        """
        async with self.lock:
            if model_id in self.models:
                print(f"Model '{model_id}' already loaded")
                return
            
            print(f"Loading model '{model_id}' from {model_path}...")
            
            # Run blocking I/O in thread pool
            await asyncio.to_thread(self._load_model_sync, model_id, model_path, task_type, device)
            
            print(f"✅ Model '{model_id}' loaded successfully")
    
    def _load_model_sync(self, model_id: str, model_path: str, task_type: str, device: str):
        """Synchronous model loading (run in thread pool)"""
        
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Normalize task type
        task_type = task_type.lower()
        if task_type not in self.TASK_MODEL_MAPPING and task_type not in ["summarization", "translation"]:
            raise ValueError(
                f"Unsupported task type: {task_type}. "
                f"Supported: {list(self.TASK_MODEL_MAPPING.keys())}"
            )
        
        # Determine device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        
        # Load model based on task type
        if task_type in self.TASK_MODEL_MAPPING:
            model_class = self.TASK_MODEL_MAPPING[task_type]
            model = model_class.from_pretrained(
                model_path,
                local_files_only=True
            )
        else:
            # For tasks like summarization, translation - use AutoModel
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                model_path,
                local_files_only=True
            )
        
        # Move to device
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        
        # Create pipeline
        pipe = pipeline(
            task=task_type,
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1
        )
        
        # Store model instance
        self.models[model_id] = ModelInstance(
            model_id=model_id,
            model=model,
            tokenizer=tokenizer,
            pipeline=pipe,
            task_type=task_type,
            device=device,
            loaded_at=datetime.now(),
            last_used=datetime.now()
        )
    
    def unload_model(self, model_id: str) -> None:
        """Unload a model and free memory"""
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not loaded")
        
        model_instance = self.models[model_id]
        
        # Move to CPU and delete
        if model_instance.device == "cuda":
            model_instance.model.cpu()
            torch.cuda.empty_cache()
        
        del self.models[model_id]
        print(f"🗑️  Model '{model_id}' unloaded")
    
    def get_model(self, model_id: str) -> ModelInstance:
        """Get a loaded model instance"""
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not loaded")
        
        model_instance = self.models[model_id]
        model_instance.last_used = datetime.now()
        return model_instance
    
    def is_model_loaded(self, model_id: str) -> bool:
        """Check if a model is loaded"""
        return model_id in self.models
    
    def get_models_info(self) -> List[Dict[str, Any]]:
        """Get information about all loaded models"""
        info_list = []
        for model_id, instance in self.models.items():
            info_list.append(self._get_instance_info(instance))
        return info_list
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not loaded")
        return self._get_instance_info(self.models[model_id])
    
    def _get_instance_info(self, instance: ModelInstance) -> Dict[str, Any]:
        """Get detailed info about a model instance"""
        memory_mb = None
        
        if instance.device == "cuda":
            try:
                memory_mb = torch.cuda.memory_allocated() / 1024**2
            except:
                pass
        
        return {
            "model_id": instance.model_id,
            "task_type": instance.task_type,
            "loaded": True,
            "device": instance.device,
            "memory_usage_mb": memory_mb,
            "loaded_at": instance.loaded_at.isoformat(),
            "last_used": instance.last_used.isoformat()
        }
    
    def cleanup(self):
        """Clean up all models"""
        print("Cleaning up all models...")
        for model_id in list(self.models.keys()):
            self.unload_model(model_id)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()