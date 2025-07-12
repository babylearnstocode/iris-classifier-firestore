import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

class ModelTrainer:
    def __init__(self):
        self.models = {
            'Decision Tree': DecisionTreeClassifier(
                criterion='gini', 
                max_depth=3, 
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=3,
                random_state=42
            ),
            'SVM': SVC(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.results = {}
        self.feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        
    def prepare_data(self, df):
        """Prepare data for training"""
        try:
            # Separate features and target
            X = df[self.feature_names]
            y = df['species']
            
            # Encode target labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Split data - sử dụng test_size=0.2 như trong Colab
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Scale features (only for SVM and Logistic Regression)
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Store original labels for internal use
            self.y_train_original = self.label_encoder.inverse_transform(y_train)
            self.y_test_original = self.label_encoder.inverse_transform(y_test)
            
            return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            st.error(f"Error preparing data: {str(e)}")
            return None, None, None, None, None, None
    
    def train_model(self, model_name, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):
        """Train a specific model"""
        try:
            model = self.models[model_name]
            
            # Use scaled data for SVM and Logistic Regression, original for tree-based models
            if model_name in ['SVM', 'Logistic Regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store results with original labels
            self.results[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_test_original': self.y_test_original,
                'y_pred_original': self.label_encoder.inverse_transform(y_pred),
                'X_train': X_train,
                'X_test': X_test,
                'X_train_scaled': X_train_scaled,
                'X_test_scaled': X_test_scaled
            }
            
            return model, accuracy
            
        except Exception as e:
            st.error(f"Error training {model_name}: {str(e)}")
            return None, 0
    
    def show_results(self, model_name):
        """Show detailed results for a model"""
        if model_name not in self.results:
            st.error(f"No results found for {model_name}")
            return
        
        result = self.results[model_name]
        
        # Show accuracy
        st.subheader(f"Results for {model_name}")
        st.metric("Accuracy", f"{result['accuracy']:.4f}")
        
        # Show confusion matrix
        cm = confusion_matrix(result['y_test'], result['y_pred'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_,
                   ax=ax)
        ax.set_title(f'Confusion Matrix - {model_name}')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        st.pyplot(fig)
        plt.close()
        
        # Show classification report
        st.subheader("Classification Report")
        report = classification_report(result['y_test'], result['y_pred'], 
                                     target_names=self.label_encoder.classes_)
        st.text(report)
        
        # Show prediction details
        st.subheader("Prediction Details")
        pred_df = pd.DataFrame({
            'Actual': result['y_test_original'],
            'Predicted': result['y_pred_original'],
            'Correct': result['y_test'] == result['y_pred']
        })
        st.dataframe(pred_df)
    
    def visualize_decision_tree(self):
        """Visualize decision tree - enhanced version"""
        if 'Decision Tree' not in self.results:
            st.error("Decision Tree model not trained yet")
            return
        
        model = self.results['Decision Tree']['model']
        
        # Enhanced decision tree visualization
        fig, ax = plt.subplots(figsize=(20, 12))
        plot_tree(model, 
                 feature_names=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
                 class_names=self.label_encoder.classes_,
                 filled=True,
                 rounded=True,
                 fontsize=12,
                 ax=ax)
        ax.set_title("Decision Tree for Iris Dataset", fontsize=16)
        st.pyplot(fig)
        plt.close()
        
        # Show tree statistics
        st.subheader("Tree Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tree Depth", model.tree_.max_depth)
        with col2:
            st.metric("Number of Nodes", model.tree_.node_count)
        with col3:
            st.metric("Number of Leaves", model.tree_.n_leaves)
    
    def show_feature_importance(self, model_name):
        """Show feature importance for tree-based models"""
        if model_name not in self.results:
            st.error(f"No results found for {model_name}")
            return
        
        model = self.results[model_name]['model']
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.subheader(f"Feature Importance - {model_name}")
            st.dataframe(importance_df)
            
            # Plot feature importance
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(importance_df['Feature'], importance_df['Importance'])
            ax.set_title(f'Feature Importance - {model_name}')
            ax.set_xlabel('Importance')
            st.pyplot(fig)
            plt.close()
        else:
            st.info(f"{model_name} does not support feature importance")
    
    def compare_models(self):
        """Compare all trained models"""
        if not self.results:
            st.info("No models trained yet")
            return
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': result['accuracy']
            })
        
        df_comparison = pd.DataFrame(comparison_data).sort_values('Accuracy', ascending=False)
        
        # Show table
        st.subheader("Model Comparison")
        st.dataframe(df_comparison)
        
        # Show bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(df_comparison['Model'], df_comparison['Accuracy'], 
                     color=['gold' if i == 0 else 'skyblue' for i in range(len(df_comparison))])
        ax.set_title('Model Accuracy Comparison')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Models')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, df_comparison['Accuracy']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    def save_model(self, model_name):
        """Save trained model"""
        if model_name not in self.results:
            st.error(f"No trained model found for {model_name}")
            return False
        
        try:
            # Create models directory
            os.makedirs("models", exist_ok=True)
            
            # Save model
            model_filename = f"models/{model_name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(self.results[model_name]['model'], model_filename)
            
            # Save scaler and encoder
            scaler_filename = f"models/{model_name.lower().replace(' ', '_')}_scaler.pkl"
            encoder_filename = f"models/{model_name.lower().replace(' ', '_')}_encoder.pkl"
            
            joblib.dump(self.scaler, scaler_filename)
            joblib.dump(self.label_encoder, encoder_filename)
            
            st.success(f"✅ Model {model_name} saved successfully!")
            st.info(f"Files saved:\n- {model_filename}\n- {scaler_filename}\n- {encoder_filename}")
            return True
            
        except Exception as e:
            st.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_name):
        """Load a saved model"""
        try:
            model_filename = f"models/{model_name.lower().replace(' ', '_')}_model.pkl"
            scaler_filename = f"models/{model_name.lower().replace(' ', '_')}_scaler.pkl"
            encoder_filename = f"models/{model_name.lower().replace(' ', '_')}_encoder.pkl"
            
            if os.path.exists(model_filename):
                model = joblib.load(model_filename)
                self.scaler = joblib.load(scaler_filename)
                self.label_encoder = joblib.load(encoder_filename)
                
                st.success(f"✅ Model {model_name} loaded successfully!")
                return model
            else:
                st.error(f"Model file not found: {model_filename}")
                return None
                
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    
    def predict_new_data(self, model_name, sepal_length, sepal_width, petal_length, petal_width):
        """Make prediction on new data"""
        if model_name not in self.results:
            st.error(f"Model {model_name} not trained yet")
            return None
        
        try:
            # Prepare input data
            X_new = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            
            model = self.results[model_name]['model']
            
            # Use scaled data for SVM and Logistic Regression
            if model_name in ['SVM', 'Logistic Regression']:
                X_new_scaled = self.scaler.transform(X_new)
                prediction = model.predict(X_new_scaled)
                probabilities = model.predict_proba(X_new_scaled) if hasattr(model, 'predict_proba') else None
            else:
                prediction = model.predict(X_new)
                probabilities = model.predict_proba(X_new) if hasattr(model, 'predict_proba') else None
            
            # Convert prediction back to original labels
            predicted_species = self.label_encoder.inverse_transform(prediction)[0]
            
            return predicted_species, probabilities
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None, None