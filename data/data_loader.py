import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st
import os
import json

class DataLoader:
    def __init__(self):
        self.dataset_path = "data/iris_dataset.csv"
        self.metadata_path = "data/dataset_metadata.json"
        
    def load_iris_from_firestore(self, firebase_config):
        """Load iris dataset from Firestore"""
        try:
            if not firebase_config:
                raise Exception("Firebase config is not available")
            
            db = firebase_config.get_firestore_client()
            if not db:
                raise Exception("Could not get Firestore client")
            
            st.info("🔄 Loading iris dataset from Firestore...")
            
            # Load dataset from Firestore collection 'iris_dataset'
            dataset_docs = db.collection('iris_dataset').stream()
            
            data = []
            for doc in dataset_docs:
                doc_data = doc.to_dict()
                data.append({
                    'sepal_length': doc_data.get('sepal_length', 0),
                    'sepal_width': doc_data.get('sepal_width', 0),
                    'petal_length': doc_data.get('petal_length', 0),
                    'petal_width': doc_data.get('petal_width', 0),
                    'species': doc_data.get('species', 'unknown')
                })
            
            if not data:
                st.warning("No data found in Firestore. Creating sample dataset...")
                return self._create_sample_dataset()
            
            df = pd.DataFrame(data)
            
            # Save dataset locally
            self._save_dataset_locally(df)
            
            # Save metadata
            metadata = {
                'source': 'firestore',
                'loaded_at': datetime.now().isoformat(),
                'total_records': len(df),
                'features': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
                'target': 'species',
                'species_counts': df['species'].value_counts().to_dict()
            }
            self._save_metadata(metadata)
            
            st.success(f"✅ Dataset loaded successfully! {len(df)} records")
            return df, metadata
            
        except Exception as e:
            st.error(f"Error loading from Firestore: {str(e)}")
            st.info("Creating sample dataset as fallback...")
            return self._create_sample_dataset()
    
    def _create_sample_dataset(self):
        """Create sample iris dataset as fallback"""
        from sklearn.datasets import load_iris
        
        # Load sklearn iris dataset
        iris = load_iris()
        
        # Create DataFrame
        df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        df['species'] = [iris.target_names[i] for i in iris.target]
        
        # Save locally
        self._save_dataset_locally(df)
        
        # Save metadata
        metadata = {
            'source': 'sklearn_fallback',
            'loaded_at': datetime.now().isoformat(),
            'total_records': len(df),
            'features': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
            'target': 'species',
            'species_counts': df['species'].value_counts().to_dict()
        }
        self._save_metadata(metadata)
        
        return df, metadata
    
    def _save_dataset_locally(self, df):
        """Save dataset to local CSV file"""
        try:
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            # Save to CSV
            df.to_csv(self.dataset_path, index=False)
            
        except Exception as e:
            st.error(f"Error saving dataset locally: {str(e)}")
    
    def _save_metadata(self, metadata):
        """Save dataset metadata"""
        try:
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            
            # Save metadata
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            st.error(f"Error saving metadata: {str(e)}")
    
    def load_local_dataset(self):
        """Load dataset from local file"""
        try:
            if os.path.exists(self.dataset_path):
                df = pd.read_csv(self.dataset_path)
                
                # Load metadata
                metadata = None
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'r') as f:
                        metadata = json.load(f)
                
                return df, metadata
            else:
                return None, None
                
        except Exception as e:
            st.error(f"Error loading local dataset: {str(e)}")
            return None, None
    
    def get_dataset_info(self):
        """Get basic information about the dataset"""
        df, metadata = self.load_local_dataset()
        
        if df is not None:
            info = {
                'total_records': len(df),
                'features': df.columns.tolist(),
                'species_counts': df['species'].value_counts().to_dict() if 'species' in df.columns else {},
                'loaded_at': metadata.get('loaded_at', 'Unknown') if metadata else 'Unknown',
                'source': metadata.get('source', 'Unknown') if metadata else 'Unknown'
            }
            return info
        
        return None
    
    def upload_sample_data_to_firestore(self, firebase_config):
        """Upload sample iris data to Firestore for testing"""
        try:
            from sklearn.datasets import load_iris
            
            if not firebase_config:
                raise Exception("Firebase config is not available")
            
            db = firebase_config.get_firestore_client()
            if not db:
                raise Exception("Could not get Firestore client")
            
            # Load sklearn iris dataset
            iris = load_iris()
            
            st.info("🔄 Uploading sample iris data to Firestore...")
            
            # Clear existing data
            existing_docs = db.collection('iris_dataset').stream()
            for doc in existing_docs:
                doc.reference.delete()
            
            # Upload new data
            batch = db.batch()
            for i, (features, target) in enumerate(zip(iris.data, iris.target)):
                doc_ref = db.collection('iris_dataset').document(f'sample_{i}')
                batch.set(doc_ref, {
                    'sepal_length': float(features[0]),
                    'sepal_width': float(features[1]),
                    'petal_length': float(features[2]),
                    'petal_width': float(features[3]),
                    'species': iris.target_names[target],
                    'uploaded_at': datetime.now()
                })
            
            batch.commit()
            st.success(f"✅ Uploaded {len(iris.data)} records to Firestore!")
            
        except Exception as e:
            st.error(f"Error uploading to Firestore: {str(e)}")