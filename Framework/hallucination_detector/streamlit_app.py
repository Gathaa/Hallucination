import streamlit as st
import pandas as pd
import json
from pathlib import Path
import sys
import os
import subprocess
import tempfile

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def run_evaluation(model_name: str, dataset: str, num_samples: int, num_workers: int):
    """Run evaluation in a separate process."""
    try:
        # Create a temporary file for the results
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name.replace('\\', '\\\\')
            # Run the evaluation script
            cmd = [
                sys.executable,
                "-c",
                f"""
import sys
import os
current_dir = r'{current_dir}'
sys.path.append(current_dir)
import json
from hallucination_detector.evaluate import HallucinationEvaluator
evaluator = HallucinationEvaluator(model_name='{model_name}', num_workers={num_workers})
results = evaluator.evaluate_{dataset}(num_samples={num_samples})
with open(r'{temp_path}', 'w') as f:
    json.dump(results, f)
                """
            ]
            subprocess.run(cmd, check=True)
            
            # Read the results
            with open(temp_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error during evaluation: {str(e)}")
        return None
    finally:
        # Clean up the temporary file
        try:
            os.unlink(temp_path)
        except:
            pass

def main():
    st.title("Hallucination Detection Evaluation")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["distilgpt2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"],
        index=0
    )
    num_samples = st.sidebar.slider("Number of Samples", 10, 100, 50)
    num_workers = st.sidebar.slider("Number of Workers", 1, 16, 4)
    
    # Main content
    st.header("Dataset Evaluation")
    
    # Dataset selection
    dataset = st.selectbox(
        "Select Dataset",
        ["faitheval", "halueval", "truthfulqa", "simpleqa", "wikibio", "all"],
        index=0
    )
    
    if st.button("Run Evaluation"):
        with st.spinner("Running evaluation..."):
            if dataset == "all":
                results = {}
                for dataset_name in ["faitheval", "halueval", "truthfulqa", "simpleqa", "wikibio"]:
                    try:
                        result = run_evaluation(model_name, dataset_name, num_samples, num_workers)
                        if result:
                            results[dataset_name] = result
                    except Exception as e:
                        st.error(f"Error evaluating {dataset_name}: {str(e)}")
                        continue
            else:
                result = run_evaluation(model_name, dataset, num_samples, num_workers)
                if result:
                    results = {dataset: result}
                else:
                    return
        
        # Display results
        for dataset_name, metrics in results.items():
            st.subheader(f"Results for {dataset_name}")
            
            # Basic metrics
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Basic Metrics**")
                st.write(f"Accuracy: {metrics['accuracy']:.4f}")
                st.write(f"Precision: {metrics['precision']:.4f}")
                st.write(f"Recall: {metrics['recall']:.4f}")
                st.write(f"F1 Score: {metrics['f1']:.4f}")
                st.write(f"Total Samples: {metrics['total_samples']}")
            
            with col2:
                st.write("**Class Distribution**")
                st.write("Ground Truth:")
                for cls, count in metrics['class_distribution']['ground_truth'].items():
                    st.write(f"- Class {cls}: {count}")
                st.write("Predictions:")
                for cls, count in metrics['class_distribution']['predictions'].items():
                    st.write(f"- Class {cls}: {count}")
            
            # Hallucination-specific metrics
            st.write("**Hallucination Metrics**")
            halluc_metrics = metrics['hallucination_metrics']
            st.write(f"Refusal Rate: {halluc_metrics['refusal_rate']:.4f}")
            st.write(f"Faithfulness Score: {halluc_metrics['faithfulness_score']:.4f}")
            st.write(f"SelfCheck Score: {halluc_metrics['selfcheck_score']:.4f}")
            st.write(f"Uncertainty Score: {halluc_metrics['uncertainty_score']:.4f}")
            
            # Warnings
            if metrics['warnings']:
                st.warning("**Warnings**")
                for warning in metrics['warnings']:
                    st.write(f"- {warning}")

if __name__ == "__main__":
    main() 