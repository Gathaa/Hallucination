from typing import List, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics import roc_curve
# Make SHAP optional
try:
    from shap import Explainer
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Explanation features will be disabled.")
from collections import Counter

class HallucinationDetector:
    def __init__(self, model_name: str = "distilgpt2", class_weights: Optional[Dict[int, float]] = None):
        """Initialize the hallucination detector.
        
        Args:
            model_name: Name of the model to use
            class_weights: Optional dictionary of class weights for balancing
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Special case for Qwen models - they require specific handling for device management
        if "Qwen" in model_name:
            print(f"Loading Qwen model {model_name} on device {self.device}")
            # Force CPU for Qwen models if having device issues
            self.device = "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map=None  # Don't use device_map for Qwen
            )
            self.model = self.model.to(self.device)  # Explicitly move model to device
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Ensure padding token is properly set
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                # Use EOS token as padding token if no pad token exists
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"Set pad_token to eos_token: {self.tokenizer.pad_token}")
            else:
                # Create a new padding token if no EOS token exists
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                # Resize model embeddings to match new vocabulary size
                self.model.resize_token_embeddings(len(self.tokenizer))
                print(f"Added new pad_token: {self.tokenizer.pad_token}")
        
        self.class_weights = class_weights or {0: 1.0, 1: 1.0}
        self.prediction_threshold = 0.5  # Default threshold
        
        # Initialize SHAP explainer for interpretability if available
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
        try:
            # Prepare input
            if context:
                input_text = f"Context: {context}\nResponse: {response}\nIs this response factually consistent with the context? Answer with 'Yes' or 'No'."
            else:
                input_text = f"Response: {response}\nIs this response factually accurate? Answer with 'Yes' or 'No'."
            
            # Special handling for Qwen models
            if "Qwen" in self.model.config._name_or_path:
                # Qwen models use a different prompt format
                if context:
                    input_text = f"<|im_start|>user\nI'll provide a context and a response. Tell me if the response is factually consistent with the context.\n\nContext: {context}\nResponse: {response}\n\nIs this response factually consistent with the context? Answer with only 'Yes' or 'No'.<|im_end|>\n<|im_start|>assistant\n"
                else:
                    input_text = f"<|im_start|>user\nIs the following response factually accurate? Answer with only 'Yes' or 'No'.\n\nResponse: {response}<|im_end|>\n<|im_start|>assistant\n"
            
            # Generate response
            inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Ensure inputs are on the correct device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Record the input for debugging
            debug_info = {
                "input_text": input_text,
                "tokenized_length": inputs["input_ids"].shape[1],
                "device": str(self.device)
            }
            
            # Ensure model is on the correct device
            self.model = self.model.to(self.device)
            
            with torch.no_grad():
                # Adjust generation parameters for different models
                generation_kwargs = {
                    "return_dict_in_generate": True,
                    "output_scores": True,
                    "pad_token_id": self.tokenizer.pad_token_id
                }
                
                if "Qwen" in self.model.config._name_or_path:
                    generation_kwargs.update({
                        "max_new_tokens": 20,
                        "do_sample": True,  # Use sampling for better responses
                        "temperature": 0.7,
                        "top_p": 0.9
                    })
                else:
                    generation_kwargs.update({
                        "max_new_tokens": 5,
                        "do_sample": False
                    })
                
                # Generate output
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            # Get logits and apply class weights
            logits = outputs.scores[0]
            weighted_logits = self._apply_class_weights(logits)
            
            # Process output
            full_output_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            
            # For Qwen, extract only the assistant's response
            if "Qwen" in self.model.config._name_or_path:
                # Get the assistant's response part
                if "<|im_start|>assistant\n" in full_output_text:
                    response_part = full_output_text.split("<|im_start|>assistant\n")[1]
                    if "<|im_end|>" in response_part:
                        response_part = response_part.split("<|im_end|>")[0]
                    output_text = response_part.strip()
                else:
                    output_text = full_output_text.strip()
            else:
                output_text = full_output_text
            
            # More robust check for "Yes" - case insensitive and look for variations
            yes_variations = ["yes", "yeah", "yep", "correct", "true", "right", "affirmative"]
            no_variations = ["no", "nope", "not", "incorrect", "false", "wrong", "negative"]
            
            # Extract the first word and check against variations
            first_word = output_text.split()[0].lower() if output_text.split() else ""
            
            if any(yes_var in output_text.lower() for yes_var in yes_variations) or first_word in yes_variations:
                consistency_score = 1.0
            elif any(no_var in output_text.lower() for no_var in no_variations) or first_word in no_variations:
                consistency_score = 0.0
            else:
                # If no clear yes/no, use a default based on the content
                consistency_score = 0.5
            
            # Calculate uncertainty
            probs = torch.softmax(weighted_logits, dim=-1)
            uncertainty = 1.0 - torch.max(probs).item()
            
            # Add debug information
            debug_info.update({
                "model": self.model.config._name_or_path,
                "raw_output": full_output_text,
                "processed_output": output_text,
                "consistency_score": consistency_score,
                "uncertainty": uncertainty,
                "max_prob": torch.max(probs).item(),
                "contains_yes": any(yes_var in output_text.lower() for yes_var in yes_variations),
                "contains_no": any(no_var in output_text.lower() for no_var in no_variations),
                "first_word": first_word,
                "pad_token": self.tokenizer.pad_token
            })
            
            return {
                "consistency_score": consistency_score,
                "is_hallucination": consistency_score < self.prediction_threshold,
                "uncertainty_score": uncertainty,
                "raw_logits": logits.cpu().numpy().tolist(),
                "raw_output": output_text,
                "debug_info": debug_info
            }
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error in detect_factual_hallucination: {str(e)}\n{error_trace}")
            
            # Return a default response to avoid breaking the evaluation
            return {
                "consistency_score": 0.5,
                "is_hallucination": False,
                "uncertainty_score": 1.0,
                "raw_output": f"ERROR: {str(e)}",
                "debug_info": {
                    "error": str(e),
                    "traceback": error_trace,
                    "pad_token": getattr(self.tokenizer, "pad_token", "None")
                }
            }
    
    def explain_prediction(self, response: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Explain model's prediction using SHAP values."""
        if not SHAP_AVAILABLE:
            return {
                "error": "SHAP not available. Please install SHAP to use explanation features.",
                "shap_values": [],
                "feature_names": []
            }
            
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
        try:
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
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error in detect_faithfulness_hallucination: {str(e)}\n{error_trace}")
            
            # Return a default response to avoid breaking the evaluation
            return {
                "similarity_score": 0.5,
                "uncertainty_score": 0.5,
                "faithfulness_score": 0.5,
                "is_hallucination": False,
                "debug_info": {
                    "error": str(e),
                    "traceback": error_trace,
                    "pad_token": getattr(self.tokenizer, "pad_token", "None")
                }
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
        
        try:
            for _ in range(num_samples):
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                responses.append(generated_text)
            
            return responses
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error in _generate_multiple_responses: {str(e)}\n{error_trace}")
            # Return original response if generation fails
            return [response]
    
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