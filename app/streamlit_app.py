import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import joblib
import json
import plotly.express as px
from sklearn.datasets import load_iris
from firebase_admin import firestore
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append('data')
sys.path.append('model')

# Import custom modules  
from data_loader import DataLoader
from model_trainer import ModelTrainer
from firebase_config import FirebaseConfig


# Initialize model trainer
model_trainer = ModelTrainer()

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False

if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

if 'dataset' not in st.session_state:
    st.session_state.dataset = None

def check_iris_dataset_exists():
    """Check if iris dataset exists in data/iris_dataset.csv"""
    try:
        dataset_path = 'data/iris_dataset.csv'
        if os.path.exists(dataset_path):
            return dataset_path
        return None
    except Exception as e:
        st.error(f"Error checking dataset: {str(e)}")
        return None

def load_existing_iris_dataset():
    """Load existing iris dataset from data/iris_dataset.csv"""
    try:
        dataset_path = 'data/iris_dataset.csv'
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            
            # Validate dataset structure
            required_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            if all(col in df.columns for col in required_columns):
                return df
            else:
                st.warning(f"Dataset found but missing required columns")
                return None
        return None
    except Exception as e:
        st.error(f"Error loading existing dataset: {str(e)}")
        return None

def load_iris_dataset():
    """Load iris dataset from sklearn"""
    try:
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        df['species'] = iris.target_names[iris.target]
        return df
    except Exception as e:
        st.error(f"Error loading iris dataset: {str(e)}")
        return None

def save_iris_dataset(df):
    """Save iris dataset to data/iris_dataset.csv"""
    try:
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Save as CSV
        df.to_csv('data/iris_dataset.csv', index=False)
        return True
    except Exception as e:
        st.error(f"Error saving dataset: {str(e)}")
        return False

def load_trained_model(model_name):
    """Load trained model from disk"""
    try:
        model_path = f"models/{model_name.lower().replace(' ', '_')}_model.pkl"
        scaler_path = f"models/{model_name.lower().replace(' ', '_')}_scaler.pkl"
        encoder_path = f"models/{model_name.lower().replace(' ', '_')}_encoder.pkl"
        
        if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(encoder_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            encoder = joblib.load(encoder_path)
            return model, scaler, encoder
        else:
            return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

def predict_with_model(features, model, scaler, encoder, model_name):
    """Make prediction using trained model"""
    try:
        # Use scaled features for SVM and Logistic Regression
        if model_name in ['SVM', 'Logistic Regression']:
            features_processed = scaler.transform([features])
        else:
            features_processed = [features]
        
        # Make prediction
        prediction = model.predict(features_processed)[0]
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(features_processed)[0]
        else:
            # For SVM without probability=True, create dummy probabilities
            prediction_proba = np.zeros(len(encoder.classes_))
            prediction_proba[prediction] = 1.0
        
        # Convert prediction to species name
        species_name = encoder.classes_[prediction]
        
        return species_name, prediction_proba, prediction
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

def save_model_metadata(model_name, accuracy):
    """Save model metadata to JSON file"""
    try:
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        metadata = {
            'model_name': model_name,
            'accuracy': accuracy,
            'trained_at': datetime.now().isoformat(),
        }
        
        metadata_path = f"models/{model_name.lower().replace(' ', '_')}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    except Exception as e:
        st.error(f"Error saving metadata: {str(e)}")

def initialize_dataset():
    """Initialize dataset on app startup"""
    if not st.session_state.dataset_loaded:
        # Check if dataset exists
        existing_df = load_existing_iris_dataset()
        if existing_df is not None:
            st.session_state.dataset_loaded = True
            st.session_state.dataset = existing_df
            return True
    return False

# Main UI
def main():
    st.set_page_config(
        page_title="Iris ML Pipeline",
        page_icon="🌸",
        layout="wide"
    )
    
    # Initialize dataset on startup
    dataset_auto_loaded = initialize_dataset()
    
    # Header
    st.title("🌸 Iris ML Pipeline: Data Loading → Training → Prediction")
    st.markdown("---")

    firebase_config = FirebaseConfig()
    db = firebase_config.get_firestore_client()

    if db:
        st.sidebar.success("☁️ Connected to Firestore")
    else:
        st.sidebar.error("❌ Failed to connect to Firestore")
    
    # Sidebar
    with st.sidebar:
        st.header("Pipeline Status")
        
        # Dataset status
        if st.session_state.dataset_loaded:
            st.success("📊 Dataset Ready")
            if dataset_auto_loaded:
                st.info("🔄 Dataset auto-loaded from existing file")
        else:
            st.warning("📊 Dataset Not Loaded")
        
        # Model status
        if st.session_state.models_trained:
            st.success("🤖 Models Trained")
        else:
            st.warning("🤖 Models Not Trained")
        
        st.markdown("---")
        st.header("Navigation")
        
        # Navigation - conditionally show Data Loading page
        pages = []
        if not st.session_state.dataset_loaded:
            pages.append("Data Loading")
        
        pages.extend(["Model Training", "Prediction", "Analytics"])
        
        if st.session_state.dataset_loaded:
            pages.insert(0, "Dataset Overview")
        
        page = st.radio(
            "Select Page",
            pages,
            help="Navigate through the ML pipeline"
        )
    
    # Main content based on selected page
    if page == "Data Loading":
        show_data_loading_page()
    elif page == "Dataset Overview":
        show_dataset_overview_page()
    elif page == "Model Training":
        show_model_training_page()
    elif page == "Prediction":
        show_prediction_page(db)
    elif page == "Analytics":
        show_analytics_page()

def show_dataset_overview_page():
    """Dataset overview page - simple view with name and 10 samples"""
    st.header("📊 Dataset Overview")
    
    if not st.session_state.dataset_loaded:
        st.error("Dataset not loaded")
        return
    
    df = st.session_state.dataset
    
    # Dataset name and basic info
    st.subheader("Iris Dataset")
    st.write(f"**Total Records:** {len(df)} | **Features:** {len(df.columns) - 1} | **Species:** {len(df['species'].unique())}")
    
    # Show 10 samples
    st.subheader("Sample Data (First 10 Records)")
    st.dataframe(df.head(10), use_container_width=True)

    # Species distribution
    st.subheader("Species Distribution")
    species_counts = df['species'].value_counts()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.bar_chart(species_counts)
    
    with col2:
        fig_pie = px.pie(values=species_counts.values, names=species_counts.index, 
                         title="Species Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)

    colors = {'setosa': 'red', 'versicolor': 'blue', 'virginica': 'green'}

    st.subheader("📊 Scatter Plot các đặc trưng chính của Iris Dataset")

    # Vẽ biểu đồ
    colors = {'setosa': 'red', 'versicolor': 'blue', 'virginica': 'green'}

    fig, axs = plt.subplots(1, 2, figsize=(8, 3))

    # Biểu đồ 1: Petal Length vs Petal Width
    for species, color in colors.items():
        subset = df[df['species'] == species]
        axs[0].scatter(subset['petal_length'], subset['petal_width'],
                    color=color, label=species)
    axs[0].set_title("Petal Length vs Petal Width")
    axs[0].set_xlabel("Petal Length (cm)")
    axs[0].set_ylabel("Petal Width (cm)")
    axs[0].legend()

    # Biểu đồ 2: Sepal Length vs Sepal Width
    for species, color in colors.items():
        subset = df[df['species'] == species]
        axs[1].scatter(subset['sepal_length'], subset['sepal_width'],
                    color=color, label=species)
    axs[1].set_title("Sepal Length vs Sepal Width")
    axs[1].set_xlabel("Sepal Length (cm)")
    axs[1].set_ylabel("Sepal Width (cm)")
    axs[1].legend()

    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)



def show_data_loading_page():
    """Data loading page - only shown when dataset is not loaded"""
    st.header("📊 Data Loading")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Load Dataset")
        
        # Check if dataset exists
        dataset_path = 'data/iris_dataset.csv'
        if os.path.exists(dataset_path):
            st.info(f"📁 Existing dataset found at: {dataset_path}")
            
            if st.button("📂 Load Existing Dataset"):
                df = load_existing_iris_dataset()
                if df is not None:
                    st.session_state.dataset_loaded = True
                    st.session_state.dataset = df
                    st.success("✅ Existing dataset loaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to load existing dataset")
        
        st.markdown("---")
        
        # Load iris dataset
        if st.button("📁 Load Fresh Iris Dataset"):
            df = load_iris_dataset()
            if df is not None:
                st.session_state.dataset_loaded = True
                st.session_state.dataset = df
                
                # Save the dataset
                if save_iris_dataset(df):
                    st.success("✅ Fresh Iris dataset loaded and saved successfully!")
                else:
                    st.warning("Dataset loaded but failed to save to disk")
                
                st.rerun()
            else:
                st.error("❌ Failed to load iris dataset")
    
    with col2:
        st.subheader("Dataset Information")
        
        if st.session_state.dataset_loaded:
            df = st.session_state.dataset
            
            # Basic info
            st.metric("Total Records", len(df))
            st.metric("Features", len(df.columns) - 1)  # Exclude target column
            st.metric("Species", len(df['species'].unique()))
            
            # Species distribution
            st.subheader("Species Distribution")
            species_counts = df['species'].value_counts()
            st.bar_chart(species_counts)
        else:
            st.info("No dataset loaded yet")
    
    # Dataset preview
    if st.session_state.dataset_loaded:
        st.markdown("---")
        st.subheader("Dataset Preview")
        
        df = st.session_state.dataset
        
        # Show first few rows
        st.dataframe(df.head())
        
        # Dataset statistics
        st.subheader("Dataset Statistics")
        st.dataframe(df.describe())
        
        # Data visualization
        st.subheader("Data Visualization")
        
        # Feature distributions
        col1, col2 = st.columns(2)
        
        with col1:
            feature = st.selectbox("Select Feature", df.columns[:-1])
            fig_hist = px.histogram(df, x=feature, color='species', 
                                   title=f"Distribution of {feature}")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Correlation matrix
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr_matrix = df[numeric_cols].corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True, 
                                title="Feature Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)

def show_model_training_page():
    """Model training page"""
    st.header("🤖 Model Training")
    
    if not st.session_state.dataset_loaded:
        st.warning("⚠️ Please load dataset first from the Data Loading page")
        return
    
    df = st.session_state.dataset
    
    # Model selection
    st.subheader("Select Models to Train")
    
    available_models = ['Decision Tree', 'Random Forest', 'SVM', 'Logistic Regression']
    selected_models = st.multiselect(
        "Choose models to train",
        available_models,
        default=['Decision Tree'],
        help="Select one or more models to train"
    )
    
    if not selected_models:
        st.warning("Please select at least one model to train")
        return
    
    # Start training
    if st.button("🚀 Start Training", type="primary"):
        # Prepare data
        with st.spinner("Preparing data..."):
            X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = model_trainer.prepare_data(df)
        
        if X_train is not None:
            st.success("✅ Data prepared successfully!")
            
            # Train selected models
            training_results = {}
            
            for model_name in selected_models:
                st.subheader(f"Training {model_name}")
                
                with st.spinner(f"Training {model_name}..."):
                    model, accuracy = model_trainer.train_model(
                        model_name, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test
                    )
                
                if model is not None:
                    st.success(f"✅ {model_name} trained successfully!")
                    
                    # Show basic metrics
                    st.metric("Test Accuracy", f"{accuracy:.3f}")
                    
                    # Show training results
                    st.subheader(f"Training Results - {model_name}")
                    model_trainer.show_results(model_name)
                    
                    # Show feature importance for tree-based models
                    if model_name in ['Decision Tree', 'Random Forest']:
                        model_trainer.show_feature_importance(model_name)
                    
                    # Special visualization for Decision Tree
                    if model_name == 'Decision Tree':
                        st.subheader("Decision Tree Structure")
                        model_trainer.visualize_decision_tree()
                    
                    # Save model
                    if model_trainer.save_model(model_name):
                        training_results[model_name] = True
                        # Save metadata
                        save_model_metadata(model_name, accuracy)
                
                st.markdown("---")
            
            # Model comparison
            if len(selected_models) > 1:
                st.subheader("Model Comparison")
                model_trainer.compare_models()
            
            # Update session state
            if training_results:
                st.session_state.models_trained = True
                st.session_state.trained_models = list(training_results.keys())
                st.success("🎉 Training completed! You can now make predictions.")

def show_prediction_page(db):
    """Prediction page"""
    st.header("🔮 Prediction")
    
    # Check if models are trained
    available_models = []
    for model_name in ['Decision Tree', 'Random Forest', 'SVM', 'Logistic Regression']:
        model_path = f"models/{model_name.lower().replace(' ', '_')}_model.pkl"
        if os.path.exists(model_path):
            available_models.append(model_name)
    
    if not available_models:
        st.warning("⚠️ No trained models found. Please train models first.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Model Selection")
        selected_model = st.selectbox(
            "Select Model",
            available_models,
            help="Choose a trained model for prediction"
        )
        
        st.subheader("Input Features")
        st.markdown("Enter the measurements of the iris flower:")
        
        # Feature inputs
        sepal_length = st.number_input(
            "Sepal Length (cm)",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1,
            help="Length of the sepal in centimeters"
        )
        
        sepal_width = st.number_input(
            "Sepal Width (cm)",
            min_value=0.0,
            max_value=10.0,
            value=3.0,
            step=0.1,
            help="Width of the sepal in centimeters"
        )
        
        petal_length = st.number_input(
            "Petal Length (cm)",
            min_value=0.0,
            max_value=10.0,
            value=4.0,
            step=0.1,
            help="Length of the petal in centimeters"
        )
        
        petal_width = st.number_input(
            "Petal Width (cm)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Width of the petal in centimeters"
        )
        
        # Predict button
        predict_button = st.button("🔮 Predict Species", type="primary")
    
    with col2:
        st.subheader("Prediction Results")
        
        if predict_button:
            # Load model
            model, scaler, encoder = load_trained_model(selected_model)
            
            if model is not None and scaler is not None and encoder is not None:
                # Make prediction
                features = [sepal_length, sepal_width, petal_length, petal_width]
                species_name, prediction_proba, prediction_class = predict_with_model(
                    features, model, scaler, encoder, selected_model
                )
                
                if species_name is not None:
                    if db:
                        try:
                            db.collection("predictions").add({
                                "sepal_length": sepal_length,
                                "sepal_width": sepal_width,
                                "petal_length": petal_length,
                                "petal_width": petal_width,
                                "predicted_species": species_name,
                                "confidence": float(prediction_proba[prediction_class]),
                                "model": selected_model,
                                "timestamp": datetime.utcnow()
                            })
                            st.success("✅ Prediction saved to Firestore!")
                        except Exception as e:
                            st.warning(f"Could not save prediction to Firestore: {e}")
                    
                    # Display results
                    st.success(f"**Predicted Species: {species_name}**")
                    st.metric("Confidence", f"{prediction_proba[prediction_class]:.1%}")
                    
                    # Show probability distribution
                    st.subheader("Probability Distribution")
                    prob_df = pd.DataFrame({
                        'Species': encoder.classes_,
                        'Probability': prediction_proba
                    })
                    st.bar_chart(prob_df.set_index('Species'))
                    
                    # Show input summary
                    st.subheader("Input Summary")
                    input_df = pd.DataFrame({
                        'Feature': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
                        'Value (cm)': features
                    })
                    st.dataframe(input_df, use_container_width=True)
            else:
                st.error("❌ Could not load trained model")

    if db:
        st.subheader("Recent Predictions from Firestore")
        try:
            docs = db.collection("predictions").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(5).stream()
            data = [doc.to_dict() for doc in docs]
            if data:
                st.dataframe(pd.DataFrame(data))
            else:
                st.info("No predictions found in Firestore.")
        except Exception as e:
            st.error(f"Failed to load data from Firestore: {e}")

def show_analytics_page():
    """Analytics page"""
    st.header("📊 Analytics & Insights")
    
    # Model performance comparison
    st.subheader("Model Performance Overview")
    
    # Check available models
    available_models = []
    model_performances = []
    
    for model_name in ['Decision Tree', 'Random Forest', 'SVM', 'Logistic Regression']:
        metadata_path = f"models/{model_name.lower().replace(' ', '_')}_metadata.json"
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    available_models.append(model_name)
                    model_performances.append({
                        'Model': model_name,
                        'Accuracy': metadata.get('accuracy', 0),
                        'Trained At': metadata.get('trained_at', 'Unknown')
                    })
            except:
                continue
    
    if model_performances:
        # Performance comparison chart
        perf_df = pd.DataFrame(model_performances)
        
        fig = px.bar(perf_df, x='Model', y='Accuracy', 
                     title='Model Accuracy Comparison',
                     text='Accuracy')
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance table
        st.subheader("Detailed Performance Metrics")
        st.dataframe(perf_df)
        
        # Best model recommendation
        best_model = perf_df.loc[perf_df['Accuracy'].idxmax()]
        st.success(f"🏆 Best performing model: **{best_model['Model']}** with {best_model['Accuracy']:.3f} accuracy")
    
    else:
        st.info("No trained models found for analysis")

    # Additional information
    st.markdown("---")
    with st.expander("ℹ️ About This Application"):
        st.markdown("""
        This application implements a complete Machine Learning pipeline for Iris flower classification:
        
        **Pipeline Steps:**
        1. **Data Loading**: Load the iris dataset (auto-detects existing datasets)
        2. **Model Training**: Train multiple ML models with real-time visualization
        3. **Prediction**: Use trained models to predict iris species
        4. **Analytics**: Analyze model performance
        
        **Supported Models:**
        - Decision Tree: Interpretable tree-based model
        - Random Forest: Ensemble of decision trees
        - SVM: Support Vector Machine classifier
        - Logistic Regression: Linear classification model
        
        **Features:**
        - Automatic dataset detection and loading
        - Real-time training visualization
        - Model performance comparison
        - Prediction confidence scoring
        - Model persistence (save/load)
        - Dataset persistence (auto-save/load)
        """)

if __name__ == "__main__":
    main()