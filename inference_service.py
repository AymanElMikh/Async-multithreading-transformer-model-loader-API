from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import os 



MODEL_LOCAL_PATH = os.path.join(".", "models", "sentiment_model")


class TransformerService:

    def __init__(self):
        """
        Initializes the model and tokenizer from the local file system only.
        """
        print(f"Attempting to load model from local path: {MODEL_LOCAL_PATH}...")
        
        try:

            if not os.path.isdir(MODEL_LOCAL_PATH):
                raise FileNotFoundError(f"Local model directory not found: {MODEL_LOCAL_PATH}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_LOCAL_PATH, 
                local_files_only=True 
            )
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_LOCAL_PATH, 
                local_files_only=True
            )
            
            self.classifier = pipeline(
                "sentiment-analysis", 
                model=self.model,
                tokenizer=self.tokenizer
            )

            print("Model loaded successfully from local files.")

        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load model from local path.")
            print(f"Details: {e}")
            raise RuntimeError("Model loading failed. Ensure all files are in the specified local directory.")

    def predict(self, text: str):
        """
        Performs inference on the given text using the loaded model.
        """
        if not text:
            return {"error": "Input text is empty."}
        
        result = self.classifier(text)[0]
        
        return {
            "label": result['label'],
            "score": float(result['score'])
        }

inference_service = TransformerService()