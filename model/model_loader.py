import streamlit as st
import pickle
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

# Configuration
MODEL_DIR = "model/saved_models"
AVAILABLE_MODELS = {
    "Random Forest": "random_forest_model.pkl",
    "Decision Tree": "decision_tree_model.pkl", 
    "Logistic Regression": "logistic_regression_model.pkl",
    "SVM": "svm_model.pkl"
}

# Backup joblib models
AVAILABLE_MODELS_JOBLIB = {
    "Random Forest": "random_forest_model_joblib.pkl",
    "Decision Tree": "decision_tree_model_joblib.pkl", 
    "Logistic Regression": "logistic_regression_model_joblib.pkl",
    "SVM": "svm_model_joblib.pkl"
}

CLASS_NAMES = {
    0: "Setosa",
    1: "Versicolor", 
    2: "Virginica"
}

class ModelLoader:
    """Class để xử lý việc load model ML"""
    
    def __init__(self):
        self.model_dir = MODEL_DIR
        self.available_models = AVAILABLE_MODELS
        self.available_models_joblib = AVAILABLE_MODELS_JOBLIB
        self.class_names = CLASS_NAMES
    
    @st.cache_resource
    def load_model_and_scaler(_self, model_name):
        """Load the selected model and scaler with enhanced compatibility"""
        try:
            model_path = os.path.join(_self.model_dir, _self.available_models[model_name])
            scaler_path = os.path.join(_self.model_dir, "scaler.pkl")
            
            model = None
            scaler = None
            
            # Method 1: Try joblib first (more compatible)
            try:
                import joblib
                model = joblib.load(model_path)
                st.success(f"Model loaded with joblib: {model_name}")
            except Exception as e1:
                # Method 2: Try joblib backup file
                try:
                    import joblib
                    joblib_path = os.path.join(_self.model_dir, _self.available_models_joblib[model_name])
                    model = joblib.load(joblib_path)
                    st.success(f"Model loaded with joblib backup: {model_name}")
                except Exception as e2:
                    # Method 3: Try pickle with different protocols
                    try:
                        # Try with pickle protocol 2 for better compatibility
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                        st.success(f"Model loaded with pickle: {model_name}")
                    except Exception as e3:
                        # Method 4: Try with pickle5 for Python 3.8+ compatibility (optional)
                        try:
                            import pickle5 as pickle_alt
                            with open(model_path, 'rb') as f:
                                model = pickle_alt.load(f)
                            st.success(f"Model loaded with pickle5: {model_name}")
                        except (Exception, ImportError) as e4:
                            st.error(f"All loading methods failed for model. Errors:")
                            st.error(f"Joblib original: {str(e1)}")
                            st.error(f"Joblib backup: {str(e2)}")
                            st.error(f"Pickle: {str(e3)}")
                            if 'ImportError' not in str(e4):
                                st.error(f"Pickle5: {str(e4)}")
                            else:
                                st.info("Pickle5 not installed - install with: pip install pickle5")
                            return None, None
            
            # Load scaler with similar approach
            scaler = _self._load_scaler(scaler_path)
            
            return model, scaler
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            _self._show_loading_solutions()
            return None, None
    
    def _load_scaler(self, scaler_path):
        """Load scaler with multiple methods"""
        try:
            import joblib
            return joblib.load(scaler_path)
        except Exception as e1:
            try:
                import joblib
                scaler_joblib_path = os.path.join(self.model_dir, "scaler_joblib.pkl")
                return joblib.load(scaler_joblib_path)
            except Exception as e2:
                try:
                    with open(scaler_path, 'rb') as f:
                        return pickle.load(f)
                except Exception as e3:
                    try:
                        import pickle5 as pickle_alt
                        with open(scaler_path, 'rb') as f:
                            return pickle_alt.load(f)
                    except Exception as e4:
                        st.error(f"All loading methods failed for scaler. Errors:")
                        st.error(f"Joblib original: {str(e1)}")
                        st.error(f"Joblib backup: {str(e2)}")
                        st.error(f"Pickle: {str(e3)}")
                        st.error(f"Pickle5: {str(e4)}")
                        return None
    
    def _show_loading_solutions(self):
        """Show possible solutions for loading issues"""
        st.info("💡 Possible solutions:")
        st.info("1. Retrain models with current Python/scikit-learn version")
        st.info("2. Install pickle5: pip install pickle5")
        st.info("3. Use joblib instead of pickle for saving models")
        st.info("4. Check if model files exist and are not corrupted")
        st.info("5. Ensure Python version compatibility")
    
    def create_dummy_models(self):
        """Create dummy models for testing when real models fail to load"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.datasets import load_iris
        
        # Load iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Create and train models
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42),
            "SVM": SVC(probability=True, random_state=42)
        }
        
        # Create scaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train models
        for name, model in models.items():
            model.fit(X_scaled, y)
        
        return models, scaler
    
    @st.cache_resource
    def get_fallback_model_and_scaler(_self, model_name):
        """Get fallback models if file loading fails"""
        try:
            models, scaler = _self.create_dummy_models()
            return models[model_name], scaler
        except Exception as e:
            st.error(f"Error creating fallback model: {str(e)}")
            return None, None
    
    @st.cache_data
    def load_metadata(_self):
        """Load training metadata"""
        try:
            metadata_path = os.path.join(_self.model_dir, "metadata.json")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            st.warning(f"Could not load metadata: {str(e)}")
            return None
    
    def predict_iris(self, features, model, scaler):
        """Make prediction on iris features"""
        try:
            # Scale the features
            features_scaled = scaler.transform([features])
            
            # Convert to DataFrame with proper column names to avoid warning
            import pandas as pd
            feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            features_df = pd.DataFrame(features_scaled, columns=feature_names)
            
            # Make prediction
            prediction = model.predict(features_df)[0]
            prediction_proba = model.predict_proba(features_df)[0]
            
            return prediction, prediction_proba
        except Exception as e:
            # Fallback to numpy array if DataFrame fails
            try:
                prediction = model.predict(features_scaled)[0]
                prediction_proba = model.predict_proba(features_scaled)[0]
                return prediction, prediction_proba
            except Exception as e2:
                st.error(f"Error making prediction: {str(e2)}")
                return None, None
    
    def get_available_models(self):
        """Get list of available models"""
        return list(self.available_models.keys())
    
    def get_class_names(self):
        """Get class names mapping"""
        return self.class_names