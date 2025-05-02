import torch
from detector import HallucinationDetector

def get_device():
    """Get the appropriate device for PyTorch."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def initialize_detector(model_name: str):
    """Initialize the hallucination detector with proper error handling."""
    try:
        device = get_device()
        detector = HallucinationDetector(model_name=model_name)
        return detector
    except Exception as e:
        print(f"Error initializing detector: {str(e)}")
        return None 