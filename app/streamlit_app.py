import streamlit as st
import pandas as pd
from datetime import datetime
import sys

# Add directories to the path to import modules
sys.path.append('data')
sys.path.append('model')

# Import custom modules  
from model_loader import ModelLoader

# Try to import firebase config
try:
    from firebase_config import FirebaseConfig
    from firebase_admin import firestore
    FIRESTORE_AVAILABLE = True
except ImportError:
    FIRESTORE_AVAILABLE = False
    st.warning("Firebase module not available. Predictions will not be saved to cloud.")

# Initialize model loader
model_loader = ModelLoader()

# Initialize Firebase config
@st.cache_resource
def get_firebase_config():
    """Get Firebase configuration instance"""
    if FIRESTORE_AVAILABLE:
        try:
            config = FirebaseConfig()
            if config.initialize_app():
                return config
            else:
                st.error("Failed to initialize Firebase")
                return None
        except Exception as e:
            st.error(f"Error initializing Firebase: {str(e)}")
            return None
    return None

def load_recent_predictions(limit=10):
    """Load recent predictions from Firestore"""
    firebase_config = get_firebase_config()
    if firebase_config:
        try:
            db = firebase_config.get_firestore_client()
            if db:
                predictions = db.collection('predictions').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(limit).stream()
                
                data = []
                for doc in predictions:
                    doc_data = doc.to_dict()
                    data.append({
                        'Timestamp': doc_data.get('timestamp', ''),
                        'Species': doc_data.get('prediction', ''),
                        'Model': doc_data.get('model_used', ''),
                        'Confidence': f"{doc_data.get('confidence', 0):.1%}",
                        'Sepal Length': doc_data.get('features', {}).get('sepal_length', 0),
                        'Sepal Width': doc_data.get('features', {}).get('sepal_width', 0),
                        'Petal Length': doc_data.get('features', {}).get('petal_length', 0),
                        'Petal Width': doc_data.get('features', {}).get('petal_width', 0)
                    })
                
                return pd.DataFrame(data)
        except Exception as e:
            st.error(f"Error loading predictions: {str(e)}")
    return pd.DataFrame()

def save_prediction_to_firestore(prediction_data):
    """Save prediction data to Firestore"""
    firebase_config = get_firebase_config()
    if firebase_config:
        try:
            db = firebase_config.get_firestore_client()
            if db:
                # Save to predictions collection
                doc_ref = db.collection('predictions').add(prediction_data)
                return doc_ref[1].id  # Return document ID
            else:
                raise Exception("Could not get Firestore client")
        except Exception as e:
            st.error(f"Error saving to Firestore: {str(e)}")
            return None
    return None

def save_prediction_result(features, prediction, model_name, confidence):
    """Save prediction result to Firestore"""
    if FIRESTORE_AVAILABLE:
        try:
            class_names = model_loader.get_class_names()
            prediction_data = {
                'timestamp': datetime.now(),
                'features': {
                    'sepal_length': features[0],
                    'sepal_width': features[1],
                    'petal_length': features[2],
                    'petal_width': features[3]
                },
                'prediction': class_names[prediction],
                'prediction_class': int(prediction),
                'model_used': model_name,
                'confidence': float(confidence),
                'user_session': st.session_state.get('session_id', 'unknown')
            }
            
            doc_id = save_prediction_to_firestore(prediction_data)
            return doc_id is not None
        except Exception as e:
            st.error(f"Error saving to Firestore: {str(e)}")
            return False
    return False

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# Main UI
def main():
    st.set_page_config(
        page_title="Iris Flower Prediction",
        page_icon="🌸",
        layout="wide"
    )
    
    # Header
    st.title("🌸 Iris Flower Species Prediction")
    st.markdown("---")
    
    # Sidebar for model selection and info
    with st.sidebar:
        # Firebase connection status
        st.header("Firebase Status")
        firebase_config = get_firebase_config()
        if firebase_config:
            st.success("🔥 Firebase Connected")
            if st.button("Test Connection"):
                if firebase_config.test_connection():
                    st.success("✅ Connection test passed!")
                else:
                    st.error("❌ Connection test failed!")
        else:
            st.error("🔥 Firebase Disconnected")
            st.info("Check firebase-config.json or environment variables")
        
        st.header("Model Configuration")
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model",
            model_loader.get_available_models(),
            help="Choose the machine learning model for prediction"
        )
        
        # Add option to use fallback models
        use_fallback = st.checkbox(
            "Use fallback models (if file loading fails)",
            value=False,
            help="Use newly trained models if saved models fail to load"
        )
        
        # Load model info
        metadata = model_loader.load_metadata()
        if metadata and selected_model.lower().replace(' ', '_') in metadata:
            model_info = metadata[selected_model.lower().replace(' ', '_')]
            st.subheader("Model Performance")
            st.metric("Accuracy", f"{model_info.get('accuracy', 'N/A'):.3f}")
            st.metric("Precision", f"{model_info.get('precision', 'N/A'):.3f}")
            st.metric("Recall", f"{model_info.get('recall', 'N/A'):.3f}")
            st.metric("F1-Score", f"{model_info.get('f1_score', 'N/A'):.3f}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
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
            # Prepare features
            features = [sepal_length, sepal_width, petal_length, petal_width]
            
            # Load model and scaler
            model, scaler = model_loader.load_model_and_scaler(selected_model)
            
            # Use fallback if main loading fails
            if (model is None or scaler is None) and use_fallback:
                st.warning("Using fallback model...")
                model, scaler = model_loader.get_fallback_model_and_scaler(selected_model)
            
            if model is not None and scaler is not None:
                # Make prediction
                prediction, prediction_proba = model_loader.predict_iris(features, model, scaler)
                
                if prediction is not None:
                    # Display results
                    class_names = model_loader.get_class_names()
                    predicted_species = class_names[prediction]
                    confidence = prediction_proba[prediction]
                    
                    st.success(f"**Predicted Species: {predicted_species}**")
                    st.metric("Confidence", f"{confidence:.1%}")
                    
                    # Show probability distribution
                    st.subheader("Probability Distribution")
                    prob_df = pd.DataFrame({
                        'Species': [class_names[i] for i in range(3)],
                        'Probability': prediction_proba
                    })
                    st.bar_chart(prob_df.set_index('Species'))
                    
                    # Save to Firestore
                    if save_prediction_result(features, prediction, selected_model, confidence):
                        st.success("✅ Prediction saved to Firestore!")
                    elif FIRESTORE_AVAILABLE:
                        st.warning("⚠️ Could not save to Firestore")
                    
                    # Show input summary
                    st.subheader("Input Summary")
                    input_df = pd.DataFrame({
                        'Feature': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
                        'Value (cm)': features
                    })
                    st.dataframe(input_df, use_container_width=True)
            else:
                st.error("❌ Could not load model. Please check model files or enable fallback models.")
    
    # Recent predictions history
    st.markdown("---")
    st.subheader("📊 Recent Predictions")
    
    if st.button("🔄 Load Recent Predictions"):
        recent_predictions = load_recent_predictions()
        if not recent_predictions.empty:
            st.dataframe(recent_predictions, use_container_width=True)
        else:
            st.info("No recent predictions found or Firebase not connected.")
    
    # Display prediction statistics
    with st.expander("📈 Prediction Statistics"):
        firebase_config = get_firebase_config()
        if firebase_config:
            try:
                db = firebase_config.get_firestore_client()
                if db:
                    # Get prediction counts by species
                    predictions = db.collection('predictions').stream()
                    species_counts = {}
                    model_counts = {}
                    
                    for doc in predictions:
                        doc_data = doc.to_dict()
                        species = doc_data.get('prediction', 'Unknown')
                        model = doc_data.get('model_used', 'Unknown')
                        
                        species_counts[species] = species_counts.get(species, 0) + 1
                        model_counts[model] = model_counts.get(model, 0) + 1
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Predictions by Species:**")
                        for species, count in species_counts.items():
                            st.write(f"- {species}: {count}")
                    
                    with col2:
                        st.write("**Predictions by Model:**")
                        for model, count in model_counts.items():
                            st.write(f"- {model}: {count}")
                            
            except Exception as e:
                st.error(f"Error loading statistics: {str(e)}")
        else:
            st.info("Firebase not connected - cannot load statistics.")
    
    # Additional information
    st.markdown("---")
    with st.expander("ℹ️ About Iris Dataset"):
        st.markdown("""
        The Iris dataset contains measurements of 150 iris flowers from three different species:
        
        - **Setosa**: Typically has smaller petals and larger sepals
        - **Versicolor**: Medium-sized flowers with moderate measurements
        - **Virginica**: Generally the largest flowers with long petals
        
        **Features:**
        - **Sepal Length**: Length of the outer floral leaf
        - **Sepal Width**: Width of the outer floral leaf  
        - **Petal Length**: Length of the inner floral leaf
        - **Petal Width**: Width of the inner floral leaf
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666666;'>"
        "🌸 Iris Flower Prediction App | Built with Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()