"""
File: model/train_model.py
Mô tả: Train các ML models cho dự đoán nhãn hoa Iris
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings('ignore')

class IrisModelTrainer:
    def __init__(self, data_path="data/processed/iris_dataset.csv"):
        """
        Khởi tạo IrisModelTrainer
        
        Args:
            data_path: Đường dẫn đến file dữ liệu CSV
        """
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        self.target_names = ['setosa', 'versicolor', 'virginica']
        
    def load_data(self):
        """Tải dữ liệu từ file CSV"""
        try:
            if os.path.exists(self.data_path):
                self.data = pd.read_csv(self.data_path)
                print(f" Tải dữ liệu thành công từ {self.data_path}")
                print(f" Kích thước dữ liệu: {self.data.shape}")
                return True
            else:
                print(f" Không tìm thấy file dữ liệu: {self.data_path}")
                print(" Hãy chạy data/load_iris.py trước để tạo dữ liệu")
                return False
                
        except Exception as e:
            print(f" Lỗi tải dữ liệu: {e}")
            return False
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Chuẩn bị dữ liệu cho training"""
        if self.data is None:
            print(" Chưa có dữ liệu để chuẩn bị!")
            return False
        
        try:
            # Tách features và target
            feature_cols = ['sepal length (cm)', 'sepal width (cm)', 
                           'petal length (cm)', 'petal width (cm)']
            X = self.data[feature_cols]
            y = self.data['target']
            
            # Chia train/test
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            print(f" Chuẩn bị dữ liệu thành công!")
            print(f" Training set: {self.X_train.shape[0]} mẫu")
            print(f" Test set: {self.X_test.shape[0]} mẫu")
            return True
            
        except Exception as e:
            print(f" Lỗi chuẩn bị dữ liệu: {e}")
            return False
    
    def train_decision_tree(self, tune_params=True):
        """Train Decision Tree model"""
        print("\n TRAINING DECISION TREE...")
        
        if tune_params:
            # Hyperparameter tuning
            param_grid = {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
            
            dt = DecisionTreeClassifier(random_state=42)
            grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            
            best_dt = grid_search.best_estimator_
            print(f"🎯 Best parameters: {grid_search.best_params_}")
            
        else:
            best_dt = DecisionTreeClassifier(random_state=42, max_depth=3)
            best_dt.fit(self.X_train, self.y_train)
        
        # Đánh giá model
        train_score = best_dt.score(self.X_train, self.y_train)
        test_score = best_dt.score(self.X_test, self.y_test)
        cv_scores = cross_val_score(best_dt, self.X_train, self.y_train, cv=5)
        
        # Lưu model và kết quả
        self.models['decision_tree'] = best_dt
        self.results['decision_tree'] = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model_params': best_dt.get_params() if tune_params else None
        }
        
        print(f"✅ Train Accuracy: {train_score:.4f}")
        print(f"✅ Test Accuracy: {test_score:.4f}")
        print(f"✅ CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return best_dt
    
    def train_random_forest(self, tune_params=True):
        """Train Random Forest model"""
        print("\n TRAINING RANDOM FOREST...")
        
        if tune_params:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            
            best_rf = grid_search.best_estimator_
            print(f"🎯 Best parameters: {grid_search.best_params_}")
            
        else:
            best_rf = RandomForestClassifier(n_estimators=100, random_state=42)
            best_rf.fit(self.X_train, self.y_train)
        
        # Đánh giá model
        train_score = best_rf.score(self.X_train, self.y_train)
        test_score = best_rf.score(self.X_test, self.y_test)
        cv_scores = cross_val_score(best_rf, self.X_train, self.y_train, cv=5)
        
        # Lưu model và kết quả
        self.models['random_forest'] = best_rf
        self.results['random_forest'] = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': dict(zip(self.feature_names, best_rf.feature_importances_))
        }
        
        print(f" Train Accuracy: {train_score:.4f}")
        print(f" Test Accuracy: {test_score:.4f}")
        print(f" CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return best_rf
    
    def train_svm(self, tune_params=True):
        """Train SVM model"""
        print("\n TRAINING SVM...")
        
        # Chuẩn hóa dữ liệu cho SVM
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        if tune_params:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear', 'poly']
            }
            
            svm = SVC(random_state=42)
            grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train_scaled, self.y_train)
            
            best_svm = grid_search.best_estimator_
            print(f" Best parameters: {grid_search.best_params_}")
            
        else:
            best_svm = SVC(random_state=42, kernel='rbf')
            best_svm.fit(X_train_scaled, self.y_train)
        
        # Đánh giá model
        train_score = best_svm.score(X_train_scaled, self.y_train)
        test_score = best_svm.score(X_test_scaled, self.y_test)
        cv_scores = cross_val_score(best_svm, X_train_scaled, self.y_train, cv=5)
        
        # Lưu model và kết quả
        self.models['svm'] = best_svm
        self.results['svm'] = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'scaled': True
        }
        
        print(f" Train Accuracy: {train_score:.4f}")
        print(f" Test Accuracy: {test_score:.4f}")
        print(f" CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return best_svm
    
    def train_logistic_regression(self):
        """Train Logistic Regression model"""
        print("\n TRAINING LOGISTIC REGRESSION...")
        
        # Chuẩn hóa dữ liệu
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Train model
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train_scaled, self.y_train)
        
        # Đánh giá model
        train_score = lr.score(X_train_scaled, self.y_train)
        test_score = lr.score(X_test_scaled, self.y_test)
        cv_scores = cross_val_score(lr, X_train_scaled, self.y_train, cv=5)
        
        # Lưu model và kết quả
        self.models['logistic_regression'] = lr
        self.results['logistic_regression'] = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'scaled': True
        }
        
        print(f" Train Accuracy: {train_score:.4f}")
        print(f" Test Accuracy: {test_score:.4f}")
        print(f" CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return lr
    
    def evaluate_models(self):
        """Đánh giá và so sánh các models"""
        print("\n📊 ĐÁNH GIÁ VÀ SO SÁNH MODELS:")
        print("=" * 60)
        
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.round(4)
        
        # Sắp xếp theo test accuracy
        results_df = results_df.sort_values('test_accuracy', ascending=False)
        
        print(results_df[['train_accuracy', 'test_accuracy', 'cv_mean', 'cv_std']])
        
        # Tìm model tốt nhất
        best_model_name = results_df.index[0]
        best_model = self.models[best_model_name]
        
        print(f"\n🏆 Model tốt nhất: {best_model_name}")
        print(f"🎯 Test Accuracy: {results_df.loc[best_model_name, 'test_accuracy']:.4f}")
        
        return best_model_name, best_model
    
    def save_models(self, model_dir="model/saved_models"):
        """Lưu các models đã train"""
        try:
            os.makedirs(model_dir, exist_ok=True)
            
            # Lưu từng model
            for name, model in self.models.items():
                model_path = os.path.join(model_dir, f"{name}_model.pkl")
                joblib.dump(model, model_path)
                print(f"💾 Lưu model: {model_path}")
            
            # Lưu scaler
            scaler_path = os.path.join(model_dir, "scaler.pkl")
            joblib.dump(self.scaler, scaler_path)
            
            # Lưu kết quả
            results_path = os.path.join(model_dir, "training_results.json")
            with open(results_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            # Lưu metadata
            metadata = {
                'trained_at': datetime.now().isoformat(),
                'feature_names': self.feature_names,
                'target_names': self.target_names,
                'data_shape': self.data.shape if self.data is not None else None,
                'train_size': len(self.X_train) if self.X_train is not None else None,
                'test_size': len(self.X_test) if self.X_test is not None else None
            }
            
            metadata_path = os.path.join(model_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Lưu tất cả models thành công vào {model_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi lưu models: {e}")
            return False
    
    def generate_detailed_report(self, model_name, model):
        """Tạo báo cáo chi tiết cho một model"""
        print(f"\n📋 BÁO CÁO CHI TIẾT - {model_name.upper()}")
        print("=" * 50)
        
        # Dự đoán trên test set
        if model_name in ['svm', 'logistic_regression']:
            X_test_scaled = self.scaler.transform(self.X_test)
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(self.X_test)
        
        # Classification report
        print("\n Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=self.target_names))
        
        # Confusion matrix
        print("\n Confusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        
        # Feature importance (nếu có)
        if hasattr(model, 'feature_importances_'):
            print("\n🔍 Feature Importance:")
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(importance_df)

def main():
    """Hàm chính để chạy training"""
    print(" IRIS MODEL TRAINER")
    print("=" * 50)
    
    # Khởi tạo trainer
    trainer = IrisModelTrainer()
    
    # Tải và chuẩn bị dữ liệu
    if not trainer.load_data():
        return
    
    if not trainer.prepare_data():
        return
    
    # Train các models
    print("\n BẮT ĐẦU TRAINING MODELS...")
    
    # Decision Tree (chính)
    dt_model = trainer.train_decision_tree(tune_params=True)
    
    # Random Forest
    rf_model = trainer.train_random_forest(tune_params=True)
    
    # SVM
    svm_model = trainer.train_svm(tune_params=True)
    
    # Logistic Regression
    lr_model = trainer.train_logistic_regression()
    
    # Đánh giá và so sánh
    best_model_name, best_model = trainer.evaluate_models()
    
    # Tạo báo cáo chi tiết cho model tốt nhất
    trainer.generate_detailed_report(best_model_name, best_model)
    
    # Lưu models
    trainer.save_models()
    
    print("\n✅ HOÀN THÀNH TRAINING!")
    print(f"🏆 Model tốt nhất: {best_model_name}")
    print("💾 Tất cả models đã được lưu vào model/saved_models/")

if __name__ == "__main__":
    main()