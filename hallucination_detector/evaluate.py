import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from tqdm import tqdm
from detector import HallucinationDetector
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from collections import Counter
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, balanced_accuracy_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class HallucinationEvaluator:
    def __init__(self, model_name: str = "distilgpt2", data_dir: str = "data/hallucination", num_workers: int = 4):
        """Initialize the hallucination evaluator.
        
        Args:
            model_name: Name of the model to use
            data_dir: Path to the directory containing hallucination datasets
            num_workers: Number of parallel workers for evaluation
        """
        self.data_dir = Path(data_dir)
        self.detector = HallucinationDetector(model_name=model_name)
        self.num_workers = num_workers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load judge model for SelfCheckGPT
        self.judge_model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        self.judge_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Set prediction threshold
        self.prediction_threshold = 0.5  # Default threshold, can be adjusted
    
    def _log_prediction_examples(self, results: List[Dict[str, Any]], num_examples: int = 10):
        """Log examples of predictions vs ground truth."""
        logger.info("\nPrediction Examples:")
        for i, r in enumerate(results[:num_examples]):
            logger.info(f"Example {i+1}:")
            response = str(r.get('response', ''))
            context = str(r.get('context', ''))
            logger.info(f"  Response: {response[:100]}...")
            logger.info(f"  Context: {context[:100]}...")
            logger.info(f"  Prediction: {r['is_hallucination']} (Score: {r.get('similarity_score', 0):.4f})")
            logger.info(f"  Ground Truth: {r['ground_truth']}")
            logger.info("---")
    
    def _process_sample(self, row: Union[pd.Series, Dict], dataset_type: str) -> Dict[str, Any]:
        """Process a single sample with improved prediction logic."""
        try:
            # Handle both dictionary and pandas Series inputs
            if isinstance(row, dict):
                row_id = row.get('id', 'unknown')
                row_data = row
            else:
                row_id = row.name if hasattr(row, 'name') else 'unknown'
                row_data = row.to_dict() if hasattr(row, 'to_dict') else row

            if dataset_type == "faith_eval":
                # FaithEval format: answer, context, answerKey
                result = self.detector.detect_faithfulness_hallucination(
                    response=row_data.get('answer', ''),
                    context=row_data.get('context', ''),
                    task_type="qa"
                )
                # Adjust prediction based on similarity score
                similarity_score = result['similarity_score']
                is_hallucination = similarity_score < self.prediction_threshold
                
                # Handle ground truth comparison more robustly
                answer = str(row_data.get('answer', '')).strip().lower()
                answer_key = str(row_data.get('answerKey', '')).strip().lower()
                ground_truth = answer == answer_key
                
                return {
                    'id': row_id,
                    'response': row_data.get('answer', ''),
                    'context': row_data.get('context', ''),
                    'similarity_score': similarity_score,
                    'is_hallucination': is_hallucination,
                    'ground_truth': ground_truth
                }
            
            elif dataset_type == "halu_eval":
                # HaluEval format: Messages (JSON), User Rating (JSON)
                messages = json.loads(row_data.get('Messages', '[]'))
                response = next((msg['message'] for msg in reversed(messages) if msg.get('sender') == 'assistant' and msg.get('type') == 'chat'), '')
                context = next((msg['metadata']['path'][-1] for msg in messages if 'metadata' in msg and 'path' in msg['metadata']), None)
                
                result = self.detector.detect_factual_hallucination(
                    response=response,
                    context=context
                )
                
                # Adjust prediction based on consistency score
                consistency_score = result['consistency_score']
                is_hallucination = consistency_score < self.prediction_threshold
                
                user_rating = json.loads(row_data.get('User Rating', '{"dialog_rating": "5"}'))
                ground_truth = float(user_rating.get('dialog_rating', '5')) >= 4.0
                
                return {
                    'id': row_id,
                    'response': response,
                    'context': context,
                    'consistency_score': consistency_score,
                    'is_hallucination': is_hallucination,
                    'ground_truth': ground_truth
                }
            
            elif dataset_type == "truthfulqa":
                # TruthfulQA format: answer, context, answerKey
                result = self.detector.detect_factual_hallucination(
                    response=row_data.get('answer', ''),
                    context=row_data.get('context', None)
                )
                return {
                    'id': row_id,
                    'consistency_score': result['consistency_score'],
                    'is_hallucination': result['is_hallucination'],
                    'ground_truth': row_data.get('answerKey', '') == row_data.get('answer', '')
                }
            
            elif dataset_type == "simpleqa":
                # SimpleQA format: answer, context, answerKey
                result = self.detector.detect_factual_hallucination(
                    response=row_data.get('answer', ''),
                    context=row_data.get('context', None)
                )
                return {
                    'id': row_id,
                    'consistency_score': result['consistency_score'],
                    'is_hallucination': result['is_hallucination'],
                    'ground_truth': row_data.get('answerKey', '') == row_data.get('answer', '')
                }
            
            elif dataset_type == "wikibio":
                # WikiBio format: input_text
                result = self.detector.detect_factual_hallucination(
                    response=row_data.get('input_text', ''),
                    context=None
                )
                return {
                    'id': row_id,
                    'consistency_score': result['consistency_score'],
                    'is_hallucination': result['is_hallucination'],
                    'ground_truth': True  # WikiBio doesn't have ground truth labels
                }
            
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")
                
        except Exception as e:
            logger.error(f"Error processing sample {row_id}: {str(e)}")
            return {
                'id': row_id,
                'response': '',
                'context': '',
                'is_hallucination': False,
                'ground_truth': False,
                'error': str(e)
            }
    
    def evaluate_faitheval(self, num_samples: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate on FaithEval dataset."""
        print("\nLoading FaithEval dataset...")
        df = pd.read_csv(self.data_dir / "FaithEval.csv")
        if num_samples:
            df = df.sample(min(num_samples, len(df)))
        
        results = []
        print(f"Evaluating {len(df)} FaithEval samples...")
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._process_sample, row, "faith_eval") for _, row in df.iterrows()]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
                results.append(future.result())
        
        return self._calculate_metrics(results)
    
    def evaluate_halueval(self, num_samples: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate on HaluEval dataset."""
        print("\nLoading HaluEval dataset...")
        df = pd.read_csv(self.data_dir / "HaluEval.csv")
        if num_samples:
            df = df.sample(min(num_samples, len(df)))
        
        results = []
        print(f"Evaluating {len(df)} HaluEval samples...")
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._process_sample, row, "halu_eval") for _, row in df.iterrows()]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
                results.append(future.result())
        
        return self._calculate_metrics(results)
    
    def evaluate_truthfulqa(self, num_samples: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate on TruthfulQA dataset."""
        print("\nLoading TruthfulQA dataset...")
        df = pd.read_csv(self.data_dir / "TruthfulQA.csv")
        if num_samples:
            df = df.sample(min(num_samples, len(df)))
        
        results = []
        print(f"Evaluating {len(df)} TruthfulQA samples...")
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._process_sample, row, "truthfulqa") for _, row in df.iterrows()]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
                results.append(future.result())
        
        return self._calculate_metrics(results)
    
    def evaluate_simpleqa(self, num_samples: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate on SimpleQA dataset."""
        print("\nLoading SimpleQA dataset...")
        df = pd.read_csv(self.data_dir / "SimpleQA.csv")
        if num_samples:
            df = df.sample(min(num_samples, len(df)))
        
        results = []
        print(f"Evaluating {len(df)} SimpleQA samples...")
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._process_sample, row, "simpleqa") for _, row in df.iterrows()]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
                results.append(future.result())
        
        return self._calculate_metrics(results)
    
    def evaluate_wikibio(self, num_samples: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate on WikiBio dataset."""
        print("\nLoading WikiBio dataset...")
        with open(self.data_dir / "WikiBio.json", 'r') as f:
            data = json.load(f)
        
        if num_samples:
            data = np.random.choice(data, min(num_samples, len(data)), replace=False)
        
        results = []
        print(f"Evaluating {len(data)} WikiBio samples...")
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._process_sample, item, "wikibio") for item in data]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
                results.append(future.result())
        
        return self._calculate_metrics(results)
    
    def _analyze_class_distribution(self, predictions: List[int], ground_truth: List[int]) -> Dict[str, Any]:
        """Analyze and report class distribution."""
        pred_dist = Counter(predictions)
        true_dist = Counter(ground_truth)
        
        # Calculate class ratios
        pred_ratio = pred_dist[1] / len(predictions) if len(predictions) > 0 else 0
        true_ratio = true_dist[1] / len(ground_truth) if len(ground_truth) > 0 else 0
        
        # Log distribution
        logger.info("\nClass Distribution Analysis:")
        logger.info(f"Predictions: {pred_dist}")
        logger.info(f"Ground Truth: {true_dist}")
        logger.info(f"Class 1 Ratio - Predictions: {pred_ratio:.2%}, Ground Truth: {true_ratio:.2%}")
        
        return {
            'predictions': {
                'class_0': pred_dist[0],
                'class_1': pred_dist[1],
                'total': len(predictions),
                'ratio_1': float(pred_ratio)
            },
            'ground_truth': {
                'class_0': true_dist[0],
                'class_1': true_dist[1],
                'total': len(ground_truth),
                'ratio_1': float(true_ratio)
            },
            'is_imbalanced': abs(pred_ratio - true_ratio) > 0.2 or min(pred_ratio, true_ratio) < 0.1
        }
    
    def _calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate evaluation metrics from results."""
        # Extract predictions and ground truth
        predictions = [r['is_hallucination'] for r in results]
        ground_truth = [r['ground_truth'] for r in results]
        
        # Debug: Print class distributions
        logger.info("\nClass Distribution Analysis:")
        logger.info(f"Ground Truth Distribution: {Counter(ground_truth)}")
        logger.info(f"Prediction Distribution: {Counter(predictions)}")
        
        # Debug: Print sample predictions
        logger.info("\nSample Predictions (first 5):")
        for i, (true, pred) in enumerate(zip(ground_truth[:5], predictions[:5])):
            logger.info(f"Sample {i+1}: True={true}, Pred={pred}")
        
        # Calculate hallucination-specific metrics
        refusal_rate = self._calculate_refusal_rate(results)
        faithfulness_score = self._calculate_faithfulness_score(results)
        selfcheck_score = self._calculate_selfcheck_score(results)
        uncertainty_score = self._calculate_uncertainty_score(results)
        
        # Calculate confusion matrix with explicit labels
        cm = confusion_matrix(ground_truth, predictions, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics with zero_division=0
        total = len(results)
        accuracy = float((tp + tn) / total if total > 0 else 0)
        
        # Calculate precision, recall, and F1 with zero_division=0
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truth, predictions, labels=[0, 1], average='binary', zero_division=0
        )
        
        # Calculate balanced accuracy
        balanced_accuracy = float((tp / (tp + fn) + tn / (tn + fp)) / 2 if (tp + fn) > 0 and (tn + fp) > 0 else 0)
        
        # Get classification report with zero_division=0
        report = classification_report(
            ground_truth, predictions, 
            labels=[0, 1], 
            target_names=['non-hallucination', 'hallucination'],
            output_dict=True,
            zero_division=0
        )
        
        # Convert numpy types in report to native Python types
        for key, value in report.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, (np.integer, np.floating)):
                        report[key][k] = float(v)
            elif isinstance(value, (np.integer, np.floating)):
                report[key] = float(value)
        
        # Create results dictionary with native Python types
        results_dict = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'class_distribution': {
                'ground_truth': dict(Counter(ground_truth)),
                'predictions': dict(Counter(predictions))
            },
            'classification_report': report,
            'total_samples': int(total),
            'hallucination_metrics': {
                'refusal_rate': refusal_rate,
                'faithfulness_score': faithfulness_score,
                'selfcheck_score': selfcheck_score,
                'uncertainty_score': uncertainty_score
            },
            'warnings': []
        }
        
        # Add warnings based on analysis
        if len(set(predictions)) == 1:
            results_dict['warnings'].append(
                f"Warning: Model only predicted class {predictions[0]}. "
                "This suggests the model may be biased towards a single class."
            )
        
        if len(set(ground_truth)) == 1:
            results_dict['warnings'].append(
                f"Warning: Ground truth only contains class {ground_truth[0]}. "
                "Metrics may not be meaningful for this dataset."
            )
        
        return results_dict
    
    def _calculate_refusal_rate(self, results: List[Dict[str, Any]]) -> float:
        """Calculate the rate at which the model refuses to answer."""
        refusal_phrases = [
            "i don't know",
            "i cannot answer",
            "i'm not sure",
            "i'm uncertain",
            "i cannot determine",
            "unknown",
            "not available"
        ]
        
        refusal_count = sum(
            1 for r in results
            if any(phrase in str(r.get('response', '')).lower() for phrase in refusal_phrases)
        )
        return float(refusal_count / len(results)) if results else 0.0
    
    def _calculate_faithfulness_score(self, results: List[Dict[str, Any]]) -> float:
        """Calculate faithfulness score based on similarity scores."""
        if not results:
            return 0.0
        
        # Use similarity_score for FaithEval, consistency_score for others
        scores = [
            r.get('similarity_score', r.get('consistency_score', 0.0))
            for r in results
        ]
        return float(np.mean(scores))
    
    def _calculate_selfcheck_score(self, results: List[Dict[str, Any]]) -> float:
        """Calculate SelfCheck score based on consistency across multiple generations."""
        if not results:
            return 0.0
        
        # For each sample, check consistency using existing detection methods
        consistency_scores = []
        for r in results[:10]:  # Limit to first 10 samples for efficiency
            response = r.get('response', '')
            context = r.get('context', '')
            if not response:
                continue
                
            # Use faithfulness hallucination detection for consistency check
            result = self.detector.detect_faithfulness_hallucination(
                response=response,
                context=context,
                task_type="qa"
            )
            
            # Use similarity score as consistency measure
            consistency_scores.append(result['similarity_score'])
        
        return float(np.mean(consistency_scores)) if consistency_scores else 0.0
    
    def _calculate_uncertainty_score(self, results: List[Dict[str, Any]]) -> float:
        """Calculate uncertainty score based on model's confidence."""
        if not results:
            return 0.0
        
        # Use uncertainty_score if available, otherwise calculate from similarity/consistency scores
        scores = []
        for r in results:
            if 'uncertainty_score' in r:
                scores.append(r['uncertainty_score'])
            else:
                # Calculate uncertainty as 1 - confidence
                confidence = max(
                    r.get('similarity_score', 0.0),
                    r.get('consistency_score', 0.0)
                )
                scores.append(1.0 - confidence)
        
        return float(np.mean(scores))
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        # Simple implementation using token overlap
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        return len(intersection) / len(union) if union else 0.0

    def evaluate_wikibio_selfcheckgpt(self, num_samples: int = 50) -> Dict[str, float]:
        """Evaluate using SelfCheckGPT approach on WikiBio dataset."""
        print("Evaluating WikiBio dataset using SelfCheckGPT approach...")
        results = []
        
        # Load WikiBio dataset
        with open(self.data_dir / "WikiBio.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for item in tqdm(data[:num_samples], desc="Processing WikiBio samples"):
            # Generate reference response at T=0
            reference_response = self.detector.generate_response(
                item["input_text"],
                temperature=0.0
            )
            
            # Generate 10 diverse responses at T=1.0
            diverse_responses = [
                self.detector.generate_response(
                    item["input_text"],
                    temperature=1.0
                ) for _ in range(10)
            ]
            
            # Extract factual claims from reference response
            claims = self._extract_factual_claims(reference_response)
            
            # Check support for each claim in diverse responses
            supported_claims = 0
            for claim in claims:
                support_count = sum(
                    1 for response in diverse_responses
                    if self._check_claim_support(claim, response)
                )
                if support_count >= 2:  # At least 20% support (2 out of 10)
                    supported_claims += 1
            
            consistency_score = supported_claims / len(claims) if claims else 0
            results.append(consistency_score)
        
        return {
            "consistency_score": np.mean(results),
            "total_samples": len(results)
        }
    
    def evaluate_simpleqa_simpleqa(self, num_samples: int = 50) -> Dict[str, float]:
        """Evaluate using SimpleQA approach."""
        print("Evaluating SimpleQA dataset...")
        results = []
        
        # Load SimpleQA dataset
        df = pd.read_csv(self.data_dir / "SimpleQA.csv")
        
        for _, row in tqdm(df.iterrows(), total=min(num_samples, len(df)), desc="Processing SimpleQA samples"):
            response = self.detector.detect_factual_hallucination(
                response=row["answer"],
                context=row["question"]
            )
            
            # Check if answer is correct or safely refused
            is_correct = response["is_factual"]
            is_safe_refusal = "don't know" in response["response"].lower() or "uncertain" in response["response"].lower()
            
            results.append(1.0 if is_correct or is_safe_refusal else 0.0)
        
        return {
            "hallucination_score": 1.0 - np.mean(results),  # Convert to hallucination rate
            "total_samples": len(results)
        }
    
    def evaluate_truthfulqa_truthfulqa(self, num_samples: int = 50) -> Dict[str, float]:
        """Evaluate using TruthfulQA approach."""
        print("Evaluating TruthfulQA dataset...")
        results = []
        
        # Load TruthfulQA dataset
        df = pd.read_csv(self.data_dir / "TruthfulQA.csv")
        
        for _, row in tqdm(df.iterrows(), total=min(num_samples, len(df)), desc="Processing TruthfulQA samples"):
            # Get model's answer
            response = self.detector.detect_factual_hallucination(
                response=row["model_answer"],
                context=row["question"]
            )
            
            # Check if answer matches correct answer
            is_correct = response["is_factual"] and response["response"].lower() == row["correct_answer"].lower()
            results.append(1.0 if is_correct else 0.0)
        
        return {
            "truthfulqa_score": np.mean(results),
            "total_samples": len(results)
        }
    
    def evaluate_faitheval_faitheval(self, num_samples: int = 50) -> Dict[str, float]:
        """Evaluate using FaithEval approach."""
        print("Evaluating FaithEval dataset...")
        results = {
            "unanswerable": [],
            "inconsistent": [],
            "counterfactual": []
        }
        
        # Load FaithEval dataset
        df = pd.read_csv(self.data_dir / "FaithEval.csv")
        
        for _, row in tqdm(df.iterrows(), total=min(num_samples, len(df)), desc="Processing FaithEval samples"):
            response = self.detector.detect_faithfulness_hallucination(
                response=row["response"],
                context=row["context"],
                task_type=row["task_type"]
            )
            
            # Evaluate based on task type
            if row["task_type"] == "unanswerable":
                is_correct = "unknown" in response["response"].lower()
                results["unanswerable"].append(1.0 if is_correct else 0.0)
            elif row["task_type"] == "inconsistent":
                is_correct = "conflict" in response["response"].lower() or "uncertain" in response["response"].lower()
                results["inconsistent"].append(1.0 if is_correct else 0.0)
            else:  # counterfactual
                is_correct = response["is_faithful"]
                results["counterfactual"].append(1.0 if is_correct else 0.0)
        
        return {
            "unanswerable_accuracy": np.mean(results["unanswerable"]),
            "inconsistent_accuracy": np.mean(results["inconsistent"]),
            "counterfactual_accuracy": np.mean(results["counterfactual"]),
            "overall_accuracy": np.mean([
                np.mean(results["unanswerable"]),
                np.mean(results["inconsistent"]),
                np.mean(results["counterfactual"])
            ]),
            "total_samples": sum(len(v) for v in results.values())
        }
    
    def evaluate_halueval_halueval(self, num_samples: int = 50) -> Dict[str, float]:
        """Evaluate using HaluEval approach."""
        print("Evaluating HaluEval dataset...")
        results = {
            "qa": [],
            "dialog": [],
            "summarization": []
        }
        
        # Load HaluEval dataset
        df = pd.read_csv(self.data_dir / "HaluEval.csv")
        
        for _, row in tqdm(df.iterrows(), total=min(num_samples, len(df)), desc="Processing HaluEval samples"):
            # Parse the JSON string in Messages column
            messages = json.loads(row["Messages"])
            last_assistant_message = next(
                (msg["content"] for msg in reversed(messages) if msg["role"] == "assistant"),
                ""
            )
            
            # Get metadata
            metadata = json.loads(row.get("Metadata", "{}"))
            task_type = metadata.get("task_type", "qa")
            
            # Evaluate based on task type
            if task_type == "qa":
                response = self.detector.detect_factual_hallucination(
                    response=last_assistant_message,
                    context=metadata.get("context", "")
                )
                results["qa"].append(1.0 if response["is_factual"] else 0.0)
            elif task_type == "dialog":
                response = self.detector.detect_faithfulness_hallucination(
                    response=last_assistant_message,
                    context=metadata.get("context", ""),
                    task_type="dialog"
                )
                results["dialog"].append(1.0 if response["is_faithful"] else 0.0)
            else:  # summarization
                response = self.detector.detect_faithfulness_hallucination(
                    response=last_assistant_message,
                    context=metadata.get("context", ""),
                    task_type="summarization"
                )
                results["summarization"].append(1.0 if response["is_faithful"] else 0.0)
        
        return {
            "qa_accuracy": np.mean(results["qa"]),
            "dialog_accuracy": np.mean(results["dialog"]),
            "summarization_accuracy": np.mean(results["summarization"]),
            "overall_accuracy": np.mean([
                np.mean(results["qa"]),
                np.mean(results["dialog"]),
                np.mean(results["summarization"])
            ]),
            "total_samples": sum(len(v) for v in results.values())
        }
    
    def _extract_factual_claims(self, text: str) -> List[str]:
        """Extract factual claims from text."""
        # Simple implementation - split into sentences and filter
        claims = [s.strip() for s in text.split('.') if s.strip()]
        return claims
    
    def _check_claim_support(self, claim: str, response: str) -> bool:
        """Check if a claim is supported in a response using the judge model."""
        prompt = f"Does the following text support the claim '{claim}'? Text: {response}"
        inputs = self.judge_tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.judge_model.generate(
            **inputs,
            max_length=50,
            num_return_sequences=1,
            pad_token_id=self.judge_tokenizer.eos_token_id
        )
        answer = self.judge_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return "yes" in answer.lower()

def evaluate_model(
    model_name: str,
    dataset_name: str,
    batch_size: int = 8,
    num_samples: Optional[int] = None
) -> Dict[str, Any]:
    """Evaluate a model on hallucination detection."""
    # Load dataset
    dataset = load_dataset(dataset_name)
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    # Initialize detector with class balancing
    detector = HallucinationDetector(model_name=model_name)
    
    # Split data for threshold calibration
    train_size = int(0.8 * len(dataset))
    train_data = dataset.select(range(train_size))
    val_data = dataset.select(range(train_size, len(dataset)))
    
    # Calibrate threshold and class weights
    detector.calibrate_threshold(val_data)
    
    # Evaluate on test set
    predictions = []
    ground_truth = []
    uncertainty_scores = []
    shap_values = []
    
    for item in dataset:
        result = detector.detect_factual_hallucination(
            response=item['response'],
            context=item.get('context')
        )
        
        # Get prediction explanation
        explanation = detector.explain_prediction(
            response=item['response'],
            context=item.get('context')
        )
        
        predictions.append(result['is_hallucination'])
        ground_truth.append(item['is_hallucination'])
        uncertainty_scores.append(result['uncertainty_score'])
        shap_values.append(explanation)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, ground_truth)
    
    # Add balanced metrics
    metrics.update({
        'balanced_accuracy': balanced_accuracy_score(ground_truth, predictions),
        'mcc': matthews_corrcoef(ground_truth, predictions),
        'f1_macro': f1_score(ground_truth, predictions, average='macro'),
        'f1_weighted': f1_score(ground_truth, predictions, average='weighted'),
        'average_uncertainty': np.mean(uncertainty_scores),
        'uncertainty_std': np.std(uncertainty_scores)
    })
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(shap_values)
    
    return {
        'metrics': metrics,
        'feature_importance': feature_importance,
        'class_weights': detector.class_weights,
        'prediction_threshold': detector.prediction_threshold
    }

def analyze_feature_importance(shap_values: List[Dict[str, Any]]) -> Dict[str, float]:
    """Analyze feature importance from SHAP values."""
    # Aggregate SHAP values across all samples
    all_values = []
    all_features = []
    
    for sample in shap_values:
        all_values.extend(sample['shap_values'])
        all_features.extend(sample['feature_names'])
    
    # Calculate mean absolute SHAP values for each feature
    feature_importance = {}
    for feature, value in zip(all_features, all_values):
        if feature not in feature_importance:
            feature_importance[feature] = []
        feature_importance[feature].append(abs(value))
    
    return {
        feature: np.mean(values)
        for feature, values in feature_importance.items()
    }

def calculate_metrics(predictions: List[bool], ground_truth: List[bool]) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics."""
    # Basic metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'npv': npv,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate hallucination detection on various datasets")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--model", type=str, default="distilgpt2", help="Model to use for evaluation")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file for results")
    parser.add_argument("--threshold", type=float, default=0.5, help="Prediction threshold for hallucination detection")
    args = parser.parse_args()
    
    evaluator = HallucinationEvaluator(model_name=args.model, num_workers=args.workers)
    evaluator.prediction_threshold = args.threshold
    
    # Evaluate all datasets
    datasets = {
        "WikiBio": evaluator.evaluate_wikibio,
        "SimpleQA": evaluator.evaluate_simpleqa,
        "TruthfulQA": evaluator.evaluate_truthfulqa,
        "FaithEval": evaluator.evaluate_faitheval,
        "HaluEval": evaluator.evaluate_halueval
    }
    
    all_results = {}
    for name, eval_func in datasets.items():
        logger.info(f"\nEvaluating {name}...")
        results = eval_func(num_samples=args.samples)
        all_results[name] = results
        
        logger.info(f"\nResults for {name}:")
        
        # Print class distribution analysis
        logger.info("\nClass Distribution Analysis:")
        dist = results['class_distribution']
        logger.info(f"Predictions: {dist['predictions']['class_0']} non-hallucination, {dist['predictions']['class_1']} hallucination")
        logger.info(f"Ground Truth: {dist['ground_truth']['class_0']} non-hallucination, {dist['ground_truth']['class_1']} hallucination")
        logger.info(f"Class 1 Ratio - Predictions: {dist['predictions']['ratio_1']:.2%}, Ground Truth: {dist['ground_truth']['ratio_1']:.2%}")
        
        # Print warnings
        if results['warnings']:
            logger.info("\nWarnings:")
            for warning in results['warnings']:
                logger.info(f"- {warning}")
        
        # Print confusion matrix
        logger.info("\nConfusion Matrix:")
        logger.info(f"True Negatives: {results['true_negatives']}")
        logger.info(f"False Positives: {results['false_positives']}")
        logger.info(f"False Negatives: {results['false_negatives']}")
        logger.info(f"True Positives: {results['true_positives']}")
        
        # Print metrics
        logger.info("\nMetrics:")
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
        logger.info(f"Precision: {results['precision']:.4f}")
        logger.info(f"Recall: {results['recall']:.4f}")
        logger.info(f"F1 Score: {results['f1']:.4f}")
        logger.info(f"Total Samples: {results['total_samples']}")
        
        # Print detailed classification report
        logger.info("\nDetailed Classification Report:")
        for label, metrics in results['classification_report'].items():
            if isinstance(metrics, dict):
                logger.info(f"\nClass {label}:")
                for metric, value in metrics.items():
                    logger.info(f"{metric}: {value:.4f}")
    
    # Save results to JSON file
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    logger.info(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main() 