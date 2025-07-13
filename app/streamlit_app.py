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

# Initialize components
data_loader = DataLoader()
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

def check_saved_models():
    """Check if all models are trained and saved"""
    required_models = ['decision_tree', 'random_forest', 'svm', 'logistic_regression']
    trained_count = 0
    
    for model in required_models:
        model_path = f"model/saved_models/{model}_model.pkl"
        if os.path.exists(model_path):
            trained_count += 1
    
    # Return True only if all models are trained
    return trained_count == len(required_models)

# Nếu có model lưu sẵn → cập nhật session state
if not st.session_state.models_trained:
    if check_saved_models():
        st.session_state.models_trained = True

def load_trained_model(model_name):
    """Load trained model from disk"""
    try:
        model_path = f"model/saved_models/{model_name.lower().replace(' ', '_')}_model.pkl"
        scaler_path = f"model/saved_models/{model_name.lower().replace(' ', '_')}_scaler.pkl"
        encoder_path = f"model/saved_models/{model_name.lower().replace(' ', '_')}_encoder.pkl"
        
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
        
        metadata_path = f"model/saved_models/{model_name.lower().replace(' ', '_')}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    except Exception as e:
        st.error(f"Error saving metadata: {str(e)}")

def initialize_dataset(db):
    """Initialize dataset using DataLoader with improved logic"""
    if not st.session_state.dataset_loaded:
        # Sử dụng logic ưu tiên mới từ DataLoader
        dataset = data_loader.initialize_dataset(db)
        if dataset is not None:
            st.session_state.dataset_loaded = True
            st.session_state.dataset = dataset
            return True
        else:
            # Nếu không có dataset, hiển thị thông báo và button get data
            return False
    return st.session_state.dataset_loaded

# Main UI
def main():
    st.set_page_config(
        page_title="Iris ML Pipeline",
        page_icon="🌸",
        layout="wide"
    )
    
    # Initialize Firebase
    firebase_config = FirebaseConfig()
    db = firebase_config.get_firestore_client()
    
    # Try to initialize dataset from local only (không auto load từ Firestore nữa)
    dataset_initialized = initialize_dataset(db)

    # Header
    st.title("🌸 Iris Classifier")
    st.markdown("---")

    if db:
        st.sidebar.success("☁️ Connected to Firestore")
    else:
        st.sidebar.error("❌ Failed to connect to Firestore")

    # Nếu chưa có dataset thì chỉ hiển thị 1 trang duy nhất
    if not st.session_state.dataset_loaded:
        st.sidebar.warning("📊 Dataset Not Loaded")
        st.sidebar.info("💡 Please load dataset from Firestore")

        st.subheader("📥 Load Iris Dataset")
        st.info("No dataset is available. Please click the button below to load from Firestore.")

        if db and st.button("☁️ Load from Firestore"):
            df = data_loader.load_from_firestore(db)
            if df is not None:
                st.session_state.dataset = df
                st.session_state.dataset_loaded = True
                data_loader.save_to_local(df)
                st.success("✅ Dataset loaded successfully!")
                st.rerun()
        return  # Dừng main() tại đây nếu chưa có dataset

    # Nếu có dataset rồi → vào giao diện chính (không có "Data Loading")
    st.sidebar.success("📊 Dataset Ready")
    dataset_info = data_loader.get_dataset_info(st.session_state.dataset)
    if dataset_info:
        st.sidebar.info(f"📋 {dataset_info['total_records']} records, {dataset_info['species_count']} species")

    if st.session_state.models_trained:
        st.sidebar.success("🤖 Models Trained")
    else:
        st.sidebar.warning("🤖 Models Not Trained")

    # Navigation Pages (bỏ "Data Loading")
    pages = ["Dataset Overview", "Model Training", "Prediction"]
    page = st.sidebar.radio("📂 Select Page", pages)

    if page == "Dataset Overview":
        show_dataset_overview_page()
    elif page == "Model Training":
        show_model_training_page()
    elif page == "Prediction":
        show_prediction_page(db)

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

def show_data_loading_page(db):
    """Data loading page - cải thiện để sử dụng logic mới từ DataLoader"""
    st.header("📊 Data Management")
    
    # Hiển thị trạng thái hiện tại
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Current Data Status")
        
        # Kiểm tra local dataset
        local_exists = data_loader.check_local_dataset_exists()
        if local_exists:
            st.success("✅ Local dataset found")
        else:
            st.info("ℹ️ No local dataset found")
        
        # Kiểm tra Firestore connection
        if db:
            st.success("✅ Firestore connection active")
        else:
            st.error("❌ Firestore connection not available")
        
        # Hiển thị dataset hiện tại
        if st.session_state.dataset_loaded:
            dataset_info = data_loader.get_dataset_info(st.session_state.dataset)
            if dataset_info:
                st.info(f"📋 Current dataset: {dataset_info['total_records']} records, {dataset_info['species_count']} species")
        else:
            st.warning("⚠️ No dataset currently loaded")
    
    with col2:
        st.subheader("Quick Actions")
        
        # Auto-load button - sử dụng logic mới từ DataLoader
        if st.button("🔄 Auto-Load Dataset", type="primary"):
            dataset = data_loader.initialize_dataset(db)
            if dataset is not None:
                st.session_state.dataset_loaded = True
                st.session_state.dataset = dataset
                st.success("✅ Dataset loaded successfully!")
                st.rerun()
            else:
                st.error("❌ Could not load dataset")
    
    st.markdown("---")
    
    # Data source options
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Local Data Management")
        
        # Load local dataset
        if local_exists:
            if st.button("📂 Load Local Dataset"):
                df = data_loader.load_local_dataset()
                if df is not None:
                    st.session_state.dataset_loaded = True
                    st.session_state.dataset = df
                    st.success("✅ Local dataset loaded!")
                    st.rerun()
        else:
            st.info("No local dataset available")
        
        # Load fresh iris dataset
        if st.button("📁 Load Fresh Iris Dataset"):
            df = data_loader.load_sklearn_iris()
            if df is not None:
                st.session_state.dataset_loaded = True
                st.session_state.dataset = df
                
                # Save to local
                data_loader.save_to_local(df)
                
                # Save to Firestore if connected
                if db:
                    data_loader.save_to_firestore(df, db)
                
                st.success("✅ Fresh Iris dataset loaded and saved!")
                st.rerun()
    
    with col2:
        st.subheader("☁️ Cloud Data Management")
        
        if db:
            # Load from Firestore
            if st.button("☁️ Load from Firestore"):
                df = data_loader.load_from_firestore(db)
                if df is not None:
                    st.session_state.dataset_loaded = True
                    st.session_state.dataset = df
                    # Save to local for future use
                    data_loader.save_to_local(df)
                    st.success("✅ Dataset loaded from Firestore!")
                    st.rerun()
            
            # Refresh from Firestore
            if st.button("🔄 Refresh from Firestore"):
                df = data_loader.refresh_from_firestore(db)
                if df is not None:
                    st.session_state.dataset_loaded = True
                    st.session_state.dataset = df
                    st.success("✅ Dataset refreshed from Firestore!")
                    st.rerun()
            
            # Upload local to Firestore
            if local_exists:
                if st.button("☁️ Upload Local to Firestore"):
                    success = data_loader.upload_local_to_firestore(db)
                    if success:
                        st.success("✅ Local dataset uploaded to Firestore!")
        else:
            st.error("❌ Firestore connection not available")
            st.info("Please check your Firebase configuration")
    
    # Dataset preview section
    if st.session_state.dataset_loaded:
        st.markdown("---")
        st.subheader("Dataset Preview")
        
        df = st.session_state.dataset
        
        # Show dataset info
        dataset_info = data_loader.get_dataset_info(df)
        if dataset_info:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", dataset_info['total_records'])
            with col2:
                st.metric("Features", dataset_info['features'])
            with col3:
                st.metric("Species", dataset_info['species_count'])
        
        # Show first few rows
        st.dataframe(df.head(10), use_container_width=True)
        
        # Species distribution
        st.subheader("Species Distribution")
        species_counts = df['species'].value_counts()
        st.bar_chart(species_counts)

def show_model_training_page():
    """Model training page - train all models at once"""
    st.header("🤖 Model Training")
    
    if not st.session_state.dataset_loaded:
        st.warning("⚠️ Please load dataset first from the Data Loading page")
        return
    
    df = st.session_state.dataset
    
    # Check if models are already trained
    if st.session_state.models_trained:
        model = model_trainer.load_model('Decision Tree')
        if model:
            # Gán thủ công vào results để enable visualization
            model_trainer.results['Decision Tree'] = {
                'model': model,
                'accuracy': 0,  # hoặc load từ metadata
                'y_test': [], 'y_pred': [],
                'y_test_original': [], 'y_pred_original': [],
                'X_train': [], 'X_test': [],
                'X_train_scaled': [], 'X_test_scaled': [],
                'train_accuracy': 0,
            }


        st.info("✅ Models have been trained! Here are the results:")
        
        # Show Decision Tree results (default display)
        st.subheader("🌳 Decision Tree Results")
        
        # Load and display decision tree results
        model_trainer.show_feature_importance('Decision Tree')
        
        st.subheader("Decision Tree Structure")
        model_trainer.visualize_decision_tree()
        
        # Option to retrain
        if st.button("🔄 Retrain All Models", type="secondary"):
            st.session_state.models_trained = False
            st.rerun()
        
        return
    
    # Training section
    st.subheader("🚀 Train All Models")
    st.info("This will train all 4 models: Decision Tree, Random Forest, SVM, and Logistic Regression")
    
    # Show available models
    available_models = ['Decision Tree', 'Random Forest', 'SVM', 'Logistic Regression']
    st.write("**Models to be trained:**")
    for model in available_models:
        st.write(f"• {model}")
    
    if st.button("🚀 Start Training All Models", type="primary"):
        # Prepare data once
        with st.spinner("Preparing data..."):
            X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = model_trainer.prepare_data(df)
        
        if X_train is not None:
            st.success("✅ Data prepared successfully!")
            
            # Progress bar for training
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            trained_models = []
            total_models = len(available_models)
            
            # Train all models
            for i, model_name in enumerate(available_models):
                status_text.text(f"Training {model_name}...")
                progress_bar.progress((i) / total_models)
                
                with st.spinner(f"Training {model_name}..."):
                    model, accuracy = model_trainer.train_model(
                        model_name, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test
                    )
                
                if model is not None:
                    trained_models.append(model_name)
                    # Save model
                    if model_trainer.save_model(model_name):
                        save_model_metadata(model_name, accuracy)
                    
                else:
                    st.error(f"❌ Failed to train {model_name}")
            
            # Complete progress
            progress_bar.progress(1.0)
            status_text.text("Training completed!")
            
            if trained_models:
                st.session_state.models_trained = True
                st.session_state.trained_models = trained_models
                
                st.success(f"🎉 Successfully trained {len(trained_models)} models!")

                # Bảng tổng hợp so sánh Accuracy
                st.subheader("📊 Accuracy Comparison of All Models")

                comparison_data = []
                for model_name in trained_models:
                    result = model_trainer.results.get(model_name)
                    if result:
                        comparison_data.append({
                            'Model': model_name,
                            'Train Accuracy': round(result.get('train_accuracy', 0), 4),
                            'Test Accuracy': round(result['accuracy'], 4)
                        })

                df_comparison = pd.DataFrame(comparison_data).sort_values(by='Test Accuracy', ascending=False)
                st.dataframe(df_comparison, use_container_width=True)
                
                # Automatically show Decision Tree results
                st.subheader("🌳 Decision Tree Results")
                
                model_trainer.show_feature_importance('Decision Tree')
                
                st.subheader("Decision Tree Structure")
                model_trainer.visualize_decision_tree()
            
                
                st.info("✨ All models trained! You can now make predictions on the Prediction page.")
            else:
                st.error("❌ No models were trained successfully")
        else:
            st.error("❌ Failed to prepare data")


def show_prediction_page(db):
    """Prediction page"""
    st.header("🔮 Prediction")
    
    # Check if models are trained
    available_models = []
    for model_name in ['Decision Tree', 'Random Forest', 'SVM', 'Logistic Regression']:
        model_path = f"model/saved_models/{model_name.lower().replace(' ', '_')}_model.pkl"
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

if __name__ == "__main__":
    main()