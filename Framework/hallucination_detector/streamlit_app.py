import streamlit as st
import pandas as pd
import json
from pathlib import Path
import sys
import os
import subprocess
import tempfile
import time

# Add the current directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import directly for debugging mode
try:
    from hallucination_detector.evaluate import HallucinationEvaluator
    DIRECT_IMPORT = True
except ImportError:
    DIRECT_IMPORT = False

def run_evaluation(model_name: str, dataset: str, num_samples: int, num_workers: int, debug_mode=False):
    """Run evaluation in a separate process or directly in debug mode."""
    if debug_mode and DIRECT_IMPORT:
        try:
            # Import the evaluator directly for debugging
            from hallucination_detector.evaluate import HallucinationEvaluator
            
            # Initialize the evaluator
            evaluator = HallucinationEvaluator(model_name=model_name, num_workers=num_workers)
            
            # Load dataset based on dataset name
            if dataset == "faitheval":
                # Load the dataset
                df = pd.read_csv(Path("data/hallucination/FaithEval.csv"))
                if num_samples > 0:
                    df = df.sample(min(num_samples, len(df)))
                
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                details_expander = st.expander("Sample Details")
                
                results = []
                total = len(df)
                
                # Process each sample individually
                for i, (_, row) in enumerate(df.iterrows()):
                    # Update progress
                    progress = float(i) / total
                    progress_bar.progress(progress)
                    status_text.text(f"Processing sample {i+1}/{total}")
                    
                    # Process the sample
                    try:
                        # Call the real method for processing
                        result = evaluator._process_sample(row, "faith_eval")
                        
                        # Display detailed information in the expander
                        with details_expander:
                            st.subheader(f"Sample {i+1}")
                            st.write(f"**Response:** {result.get('response', 'N/A')[:300]}...")
                            st.write(f"**Context:** {result.get('context', 'N/A')[:300]}...")
                            st.write(f"**Prediction:** {'Hallucination' if result.get('is_hallucination', False) else 'Not Hallucination'}")
                            st.write(f"**Ground Truth:** {'Hallucination' if result.get('ground_truth', False) else 'Not Hallucination'}")
                            st.write(f"**Similarity Score:** {result.get('similarity_score', 0):.4f}")
                            st.write(f"**Uncertainty Score:** {result.get('uncertainty_score', 0):.4f}")
                            
                            # Display raw model output and debug info if available
                            if 'raw_output' in result:
                                st.write(f"**Raw Model Output:** {result['raw_output']}")
                            
                            # Display debug info if available
                            if 'debug_info' in result:
                                debug_expander = st.expander("Debug Information")
                                with debug_expander:
                                    debug_info = result['debug_info']
                                    for key, value in debug_info.items():
                                        if isinstance(value, (str, int, float, bool)):
                                            st.write(f"**{key}:** {value}")
                                        elif key == "raw_logits":
                                            st.write(f"**{key}:** [tensor data]")
                                        else:
                                            st.write(f"**{key}:** {str(value)}")
                            
                            st.write("---")
                        
                        results.append(result)
                    except Exception as e:
                        with details_expander:
                            st.error(f"Error processing sample {i+1}: {str(e)}")
                    
                    # Small delay to make UI updates visible
                    time.sleep(0.1)
                
                # Complete the progress bar
                progress_bar.progress(1.0)
                status_text.text("Processing complete!")
                
                # Calculate metrics
                return evaluator._calculate_metrics(results)
            
            elif dataset == "halueval":
                df = pd.read_csv(Path("data/hallucination/HaluEval.csv"))
                if num_samples > 0:
                    df = df.sample(min(num_samples, len(df)))
                
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                details_expander = st.expander("Sample Details")
                
                results = []
                total = len(df)
                
                # Process each sample individually
                for i, (_, row) in enumerate(df.iterrows()):
                    # Update progress
                    progress = float(i) / total
                    progress_bar.progress(progress)
                    status_text.text(f"Processing sample {i+1}/{total}")
                    
                    # Process the sample
                    try:
                        # Call the real method for processing
                        result = evaluator._process_sample(row, "halu_eval")
                        
                        # Display detailed information in the expander
                        with details_expander:
                            st.subheader(f"Sample {i+1}")
                            st.write(f"**Response:** {result.get('response', 'N/A')[:300]}...")
                            st.write(f"**Context:** {result.get('context', 'N/A')[:300]}...")
                            st.write(f"**Prediction:** {'Hallucination' if result.get('is_hallucination', False) else 'Not Hallucination'}")
                            st.write(f"**Ground Truth:** {'Hallucination' if result.get('ground_truth', False) else 'Not Hallucination'}")
                            st.write(f"**Consistency Score:** {result.get('consistency_score', 0):.4f}")
                            st.write("---")
                        
                        results.append(result)
                    except Exception as e:
                        with details_expander:
                            st.error(f"Error processing sample {i+1}: {str(e)}")
                    
                    # Small delay to make UI updates visible
                    time.sleep(0.1)
                
                # Complete the progress bar
                progress_bar.progress(1.0)
                status_text.text("Processing complete!")
                
                # Calculate metrics
                return evaluator._calculate_metrics(results)
            
            # Add similar blocks for other dataset types
            elif dataset == "truthfulqa":
                df = pd.read_csv(Path("data/hallucination/TruthfulQA.csv"))
                if num_samples > 0:
                    df = df.sample(min(num_samples, len(df)))
                
                # Create a progress bar and other UI elements
                progress_bar = st.progress(0)
                status_text = st.empty()
                details_expander = st.expander("Sample Details")
                
                results = []
                total = len(df)
                
                # Process each sample individually
                for i, (_, row) in enumerate(df.iterrows()):
                    progress = float(i) / total
                    progress_bar.progress(progress)
                    status_text.text(f"Processing sample {i+1}/{total}")
                    
                    try:
                        result = evaluator._process_sample(row, "truthfulqa")
                        
                        with details_expander:
                            st.subheader(f"Sample {i+1}")
                            st.write(f"**Question:** {row.get('Question', 'N/A')}")
                            st.write(f"**Answer:** {row.get('Best Answer', 'N/A')}")
                            st.write(f"**Prediction:** {'Hallucination' if result.get('is_hallucination', False) else 'Not Hallucination'}")
                            st.write(f"**Ground Truth:** {'Hallucination' if result.get('ground_truth', False) else 'Not Hallucination'}")
                            st.write(f"**Consistency Score:** {result.get('consistency_score', 0):.4f}")
                            st.write("---")
                        
                        results.append(result)
                    except Exception as e:
                        with details_expander:
                            st.error(f"Error processing sample {i+1}: {str(e)}")
                    
                    time.sleep(0.1)
                
                progress_bar.progress(1.0)
                status_text.text("Processing complete!")
                
                return evaluator._calculate_metrics(results)
            
            elif dataset == "simpleqa":
                df = pd.read_csv(Path("data/hallucination/SimpleQA.csv"))
                if num_samples > 0:
                    df = df.sample(min(num_samples, len(df)))
                
                # Create a progress bar and other UI elements
                progress_bar = st.progress(0)
                status_text = st.empty()
                details_expander = st.expander("Sample Details")
                
                results = []
                total = len(df)
                
                # Process each sample individually
                for i, (_, row) in enumerate(df.iterrows()):
                    progress = float(i) / total
                    progress_bar.progress(progress)
                    status_text.text(f"Processing sample {i+1}/{total}")
                    
                    try:
                        result = evaluator._process_sample(row, "simpleqa")
                        
                        with details_expander:
                            st.subheader(f"Sample {i+1}")
                            st.write(f"**Question:** {row.get('question', 'N/A')}")
                            st.write(f"**Answer:** {row.get('answer', 'N/A')}")
                            st.write(f"**Prediction:** {'Hallucination' if result.get('is_hallucination', False) else 'Not Hallucination'}")
                            st.write(f"**Ground Truth:** {'Hallucination' if result.get('ground_truth', False) else 'Not Hallucination'}")
                            st.write(f"**Consistency Score:** {result.get('consistency_score', 0):.4f}")
                            st.write("---")
                        
                        results.append(result)
                    except Exception as e:
                        with details_expander:
                            st.error(f"Error processing sample {i+1}: {str(e)}")
                    
                    time.sleep(0.1)
                
                progress_bar.progress(1.0)
                status_text.text("Processing complete!")
                
                return evaluator._calculate_metrics(results)
            
            elif dataset == "wikibio":
                # Load WikiBio JSON data
                with open(Path("data/hallucination/WikiBio.json"), 'r') as f:
                    data = json.load(f)
                
                if num_samples > 0:
                    import numpy as np
                    data = np.random.choice(data, min(num_samples, len(data)), replace=False).tolist()
                
                # Create a progress bar and other UI elements
                progress_bar = st.progress(0)
                status_text = st.empty()
                details_expander = st.expander("Sample Details")
                
                results = []
                total = len(data)
                
                # Process each sample individually
                for i, item in enumerate(data):
                    progress = float(i) / total
                    progress_bar.progress(progress)
                    status_text.text(f"Processing sample {i+1}/{total}")
                    
                    try:
                        result = evaluator._process_sample(item, "wikibio")
                        
                        with details_expander:
                            st.subheader(f"Sample {i+1}")
                            st.write(f"**Text:** {item.get('input_text', 'N/A')[:300]}...")
                            st.write(f"**Prediction:** {'Hallucination' if result.get('is_hallucination', False) else 'Not Hallucination'}")
                            st.write(f"**Ground Truth:** {'Hallucination' if result.get('ground_truth', False) else 'Not Hallucination'}")
                            st.write(f"**Consistency Score:** {result.get('consistency_score', 0):.4f}")
                            st.write("---")
                        
                        results.append(result)
                    except Exception as e:
                        with details_expander:
                            st.error(f"Error processing sample {i+1}: {str(e)}")
                    
                    time.sleep(0.1)
                
                progress_bar.progress(1.0)
                status_text.text("Processing complete!")
                
                return evaluator._calculate_metrics(results)
            
            else:
                st.error(f"Unknown dataset: {dataset}")
                return None
            
        except Exception as e:
            st.error(f"Error during evaluation: {str(e)}")
            return None
    else:
        # Use the subprocess approach for non-debug mode
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
results = evaluator.evaluate_{dataset}(num_samples={num_samples if num_samples > 0 else 'None'})
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
        ["distilgpt2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Qwen/Qwen2-0.5B-Instruct"],
        index=0
    )
    
    # Add a checkbox for using the full dataset
    use_full_dataset = st.sidebar.checkbox("Use Full Dataset", value=False, 
                                        help="If checked, evaluation will use the entire dataset")
    
    # Only show samples slider if not using full dataset
    if not use_full_dataset:
        num_samples = st.sidebar.slider("Number of Samples", 10, 100, 50)
    else:
        num_samples = -1  # Special value to indicate full dataset
        st.sidebar.info(f"Using full datasets:\n- SimpleQA: 4,326 samples\n- TruthfulQA: 816 samples\n- WikiBio: 239 samples\n- FaithEval: 4,992 samples\n- HaluEval: 10,000 samples")
    
    num_workers = st.sidebar.slider("Number of Workers", 1, 16, 4)
    
    # Add debug mode checkbox
    debug_mode = st.sidebar.checkbox("Debug Mode (Per-cell Evaluation)", value=False,
                                  help="If checked, shows detailed per-cell evaluation with progress bar")
    
    # Main content
    st.header("Dataset Evaluation")
    
    # Dataset selection
    dataset = st.selectbox(
        "Select Dataset",
        ["faitheval", "halueval", "truthfulqa", "simpleqa", "wikibio", "all"],
        index=0
    )
    
    # Disable "all" option in debug mode
    if debug_mode and dataset == "all":
        st.warning("Debug mode doesn't support evaluating all datasets at once. Please select a specific dataset.")
    
    if st.button("Run Evaluation"):
        if debug_mode and dataset == "all":
            st.error("Please select a specific dataset for debug mode.")
            return
        
        with st.spinner("Running evaluation..." if not debug_mode else "Setting up evaluation..."):
            if dataset == "all" and not debug_mode:
                results = {}
                for dataset_name in ["faitheval", "halueval", "truthfulqa", "simpleqa", "wikibio"]:
                    try:
                        result = run_evaluation(model_name, dataset_name, num_samples, num_workers, debug_mode=False)
                        if result:
                            results[dataset_name] = result
                    except Exception as e:
                        st.error(f"Error evaluating {dataset_name}: {str(e)}")
                        continue
            else:
                result = run_evaluation(model_name, dataset, num_samples, num_workers, debug_mode=debug_mode)
                if result:
                    results = {dataset: result}
                else:
                    return
        
        # Display results
        st.header("Evaluation Results")
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