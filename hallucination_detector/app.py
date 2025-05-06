def main():
    st.title("Hallucination Detection Framework")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    model_name = st.sidebar.selectbox(
        "Select Model",
        ["distilgpt2", "gpt2", "gpt2-medium", "gpt2-large"]
    )
    
    dataset_name = st.sidebar.selectbox(
        "Select Dataset",
        ["hallucination_detector/data/hallucination/FaithEval.csv",
         "hallucination_detector/data/hallucination/HaluEval.csv",
         "hallucination_detector/data/hallucination/TruthfulQA.csv"]
    )
    
    num_samples = st.sidebar.slider(
        "Number of Samples",
        min_value=10,
        max_value=1000,
        value=100,
        step=10
    )
    
    # Main content
    if st.sidebar.button("Run Evaluation"):
        with st.spinner("Running evaluation..."):
            results = evaluate_model(
                model_name=model_name,
                dataset_name=dataset_name,
                num_samples=num_samples
            )
            
            # Display metrics
            st.header("Evaluation Results")
            
            # Basic metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{results['metrics']['accuracy']:.3f}")
                st.metric("Balanced Accuracy", f"{results['metrics']['balanced_accuracy']:.3f}")
            with col2:
                st.metric("Precision", f"{results['metrics']['precision']:.3f}")
                st.metric("Recall", f"{results['metrics']['recall']:.3f}")
            with col3:
                st.metric("F1 Score", f"{results['metrics']['f1']:.3f}")
                st.metric("MCC", f"{results['metrics']['mcc']:.3f}")
            
            # Confusion matrix
            st.subheader("Confusion Matrix")
            cm_data = {
                'True Negatives': results['metrics']['true_negatives'],
                'False Positives': results['metrics']['false_positives'],
                'False Negatives': results['metrics']['false_negatives'],
                'True Positives': results['metrics']['true_positives']
            }
            st.bar_chart(cm_data)
            
            # Uncertainty analysis
            st.subheader("Uncertainty Analysis")
            st.metric("Average Uncertainty", f"{results['metrics']['average_uncertainty']:.3f}")
            st.metric("Uncertainty Std", f"{results['metrics']['uncertainty_std']:.3f}")
            
            # Feature importance
            st.subheader("Feature Importance")
            feature_importance = results['feature_importance']
            top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
            st.bar_chart(top_features)
            
            # Model configuration
            st.subheader("Model Configuration")
            st.write(f"Prediction Threshold: {results['prediction_threshold']:.3f}")
            st.write("Class Weights:", results['class_weights'])
            
            # Sample predictions
            st.subheader("Sample Predictions")
            sample_data = load_dataset(dataset_name).select(range(5))
            for idx, item in enumerate(sample_data):
                with st.expander(f"Sample {idx + 1}"):
                    st.write("Response:", item['response'])
                    if 'context' in item:
                        st.write("Context:", item['context'])
                    result = detect_factual_hallucination(
                        response=item['response'],
                        context=item.get('context')
                    )
                    st.write("Prediction:", "Hallucination" if result['is_hallucination'] else "Not Hallucination")
                    st.write("Confidence:", f"{1 - result['uncertainty_score']:.3f}")
                    
                    # SHAP explanation
                    explanation = explain_prediction(
                        response=item['response'],
                        context=item.get('context')
                    )
                    st.write("Key Features:", explanation['feature_names'][:5]) 