from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics import roc_curve
from shap import Explainer
from collections import Counter

class HallucinationDetector:
    def __init__(self, model_name: str = "distilgpt2", class_weights: Optional[Dict[int, float]] = None):
        """Initialize the hallucination detector.
        
        Args:
            model_name: Name of the model to use
            class_weights: Optional dictionary of class weights for balancing
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.class_weights = class_weights or {0: 1.0, 1: 1.0}
        self.prediction_threshold = 0.5  # Default threshold
        
        # Initialize SHAP explainer for interpretability
        self.explainer = None
        
    def _calculate_class_weights(self, labels: List[int]) -> Dict[int, float]:
        """Calculate class weights based on label distribution."""
        label_counts = Counter(labels)
        total = sum(label_counts.values())
        return {
            label: total / (len(label_counts) * count)
            for label, count in label_counts.items()
        }
    
    def _optimize_threshold(self, scores: List[float], labels: List[int]) -> float:
        """Optimize prediction threshold using ROC curve."""
        fpr, tpr, thresholds = roc_curve(labels, scores)
        # Find threshold that maximizes TPR - FPR
        optimal_idx = np.argmax(tpr - fpr)
        return thresholds[optimal_idx]
    
    def _apply_class_weights(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply class weights to logits."""
        weights = torch.tensor([self.class_weights[0], self.class_weights[1]], 
                             device=self.device)
        return logits * weights
    
    def detect_factual_hallucination(self, response: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Detect factual hallucinations in a response."""
        # Prepare input
        if context:
            input_text = f"Context: {context}\nResponse: {response}\nIs this response factually consistent with the context? Answer with 'Yes' or 'No'."
        else:
            input_text = f"Response: {response}\nIs this response factually accurate? Answer with 'Yes' or 'No'."
        
        # Generate response
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Get logits and apply class weights
        logits = outputs.scores[0]
        weighted_logits = self._apply_class_weights(logits)
        
        # Process output
        output_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        consistency_score = 1.0 if "Yes" in output_text else 0.0
        
        # Calculate uncertainty
        probs = torch.softmax(weighted_logits, dim=-1)
        uncertainty = 1.0 - torch.max(probs).item()
        
        return {
            "consistency_score": consistency_score,
            "is_hallucination": consistency_score < self.prediction_threshold,
            "uncertainty_score": uncertainty,
            "raw_logits": logits.cpu().numpy().tolist()
        }
    
    def explain_prediction(self, response: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Explain model's prediction using SHAP values."""
        if self.explainer is None:
            self.explainer = Explainer(self.model, self.tokenizer)
        
        # Prepare input
        if context:
            input_text = f"Context: {context}\nResponse: {response}\nIs this response factually consistent with the context?"
        else:
            input_text = f"Response: {response}\nIs this response factually accurate?"
        
        # Get SHAP values
        shap_values = self.explainer([input_text])
        
        return {
            "shap_values": shap_values.values.tolist(),
            "feature_names": self.tokenizer.tokenize(input_text)
        }
    
    def calibrate_threshold(self, validation_data: List[Dict[str, Any]]) -> None:
        """Calibrate prediction threshold using validation data."""
        scores = []
        labels = []
        
        for item in validation_data:
            result = self.detect_factual_hallucination(
                response=item['response'],
                context=item.get('context')
            )
            scores.append(result['consistency_score'])
            labels.append(1 if item['is_hallucination'] else 0)
        
        self.prediction_threshold = self._optimize_threshold(scores, labels)
        self.class_weights = self._calculate_class_weights(labels)
    
    def detect_faithfulness_hallucination(self, response: str, context: str, task_type: str) -> Dict[str, Any]:
        """Detect faithfulness hallucinations in a response.
        
        Args:
            response: The response to evaluate
            context: The context to compare against
            task_type: Type of task (e.g., "summarization", "translation")
            
        Returns:
            Dictionary containing:
            - similarity_score: Score indicating similarity to context
            - uncertainty_score: Score indicating model's uncertainty
            - faithfulness_score: Combined score
            - is_hallucination: Boolean indicating if hallucination is detected
        """
        # Prepare input for similarity check
        similarity_input = f"Context: {context}\nResponse: {response}\nHow similar is the response to the context? Answer with a number between 0 and 1."
        
        # Generate similarity score
        inputs = self.tokenizer(similarity_input, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False
            )
        
        # Process similarity score
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            similarity_score = float(output_text.split()[-1])
            similarity_score = max(0.0, min(1.0, similarity_score))
        except (ValueError, IndexError):
            similarity_score = 0.5
        
        # Prepare input for uncertainty check
        uncertainty_input = f"Response: {response}\nHow confident are you about the accuracy of this response? Answer with a number between 0 and 1."
        
        # Generate uncertainty score
        inputs = self.tokenizer(uncertainty_input, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=False
            )
        
        # Process uncertainty score
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            uncertainty_score = float(output_text.split()[-1])
            uncertainty_score = max(0.0, min(1.0, uncertainty_score))
        except (ValueError, IndexError):
            uncertainty_score = 0.5
        
        # Calculate faithfulness score
        faithfulness_score = (similarity_score + uncertainty_score) / 2
        
        return {
            "similarity_score": similarity_score,
            "uncertainty_score": uncertainty_score,
            "faithfulness_score": faithfulness_score,
            "is_hallucination": faithfulness_score < 0.5
        }
    
    def _generate_multiple_responses(
        self,
        response: str,
        temperature: float,
        num_samples: int
    ) -> List[str]:
        """Generate multiple responses using the selected model."""
        prompt = f"Please rephrase the following response in different ways: {response}"
        responses = []
        
        for _ in range(num_samples):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(generated_text)
        
        return responses
    
    def _calculate_consistency_score(
        self,
        original_response: str,
        other_responses: List[str]
    ) -> float:
        """Calculate consistency score using semantic similarity."""
        scores = []
        for response in other_responses:
            score = self._calculate_semantic_similarity(original_response, response)
            scores.append(score)
        return np.mean(scores)
    
    def _calculate_semantic_similarity(
        self,
        response: str,
        context: str
    ) -> float:
        """Calculate semantic similarity between response and context."""
        response_embedding = self.similarity_model.encode(response)
        context_embedding = self.similarity_model.encode(context)
        return np.dot(response_embedding, context_embedding) / (
            np.linalg.norm(response_embedding) * np.linalg.norm(context_embedding)
        )
    
    def _check_uncertainty_acknowledgment(self, response: str) -> float:
        """Check if response acknowledges uncertainty."""
        uncertainty_phrases = [
            "i don't know",
            "i cannot answer",
            "i'm not sure",
            "i'm uncertain",
            "i cannot determine"
        ]
        return 1.0 if any(phrase in response.lower() for phrase in uncertainty_phrases) else 0.0
    
    def _calculate_faithfulness_score(
        self,
        response: str,
        context: str,
        task_type: str
    ) -> float:
        """Calculate faithfulness score based on task type."""
        if task_type == "qa":
            return self._calculate_qa_faithfulness(response, context)
        elif task_type == "summarization":
            return self._calculate_summarization_faithfulness(response, context)
        else:
            return self._calculate_semantic_similarity(response, context)
    
    def _calculate_qa_faithfulness(self, response: str, context: str) -> float:
        """Calculate faithfulness score for QA tasks."""
        return self._calculate_semantic_similarity(response, context)
    
    def _calculate_summarization_faithfulness(self, response: str, context: str) -> float:
        """Calculate faithfulness score for summarization tasks."""
        return self._calculate_semantic_similarity(response, context) 