"""
File: data/load_iris.py
Mô tả: Tải và xử lý dữ liệu Iris dataset, lưu vào Firebase
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import firebase_admin
from firebase_admin import credentials, firestore
import os
from datetime import datetime
import json
import sys
sys.path.append('..')
from firebase_config import FirebaseConfig

class IrisDataLoader:
    def __init__(self):
        """Khởi tạo IrisDataLoader"""
        self.firebase_config = FirebaseConfig()
        self.db = None
        self.iris_data = None
        
    def initialize_firebase(self):
        """Khởi tạo kết nối Firebase"""
        if self.firebase_config.initialize_app():
            self.db = self.firebase_config.get_firestore_client()
            return True
        return False
    
    def load_iris_dataset(self):
        """Tải dữ liệu Iris từ sklearn"""
        try:
            # Tải dữ liệu Iris
            iris = load_iris()
            
            # Tạo DataFrame
            df = pd.DataFrame(iris.data, columns=iris.feature_names)
            df['target'] = iris.target
            df['target_name'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
            
            # Làm tròn các giá trị đo lường
            numeric_columns = ['sepal length (cm)', 'sepal width (cm)', 
                             'petal length (cm)', 'petal width (cm)']
            df[numeric_columns] = df[numeric_columns].round(2)
            
            self.iris_data = df
            print(f"✅ Tải dữ liệu Iris thành công! Tổng cộng {len(df)} mẫu")
            
            return df
            
        except Exception as e:
            print(f"❌ Lỗi tải dữ liệu Iris: {e}")
            return None
    
    def save_to_csv(self, file_path="data/processed/iris_dataset.csv"):
        """Lưu dữ liệu ra file CSV"""
        try:
            if self.iris_data is None:
                print("❌ Chưa có dữ liệu để lưu!")
                return False
            
            # Tạo thư mục nếu chưa tồn tại
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Lưu file
            self.iris_data.to_csv(file_path, index=False)
            print(f"✅ Lưu dữ liệu CSV thành công: {file_path}")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi lưu file CSV: {e}")
            return False
    
    def upload_to_firebase(self, collection_name="iris_dataset"):
        """Upload dữ liệu lên Firebase Firestore"""
        if self.db is None:
            print("❌ Chưa kết nối Firebase!")
            return False
        
        if self.iris_data is None:
            print("❌ Chưa có dữ liệu để upload!")
            return False
        
        try:
            # Xóa collection cũ (nếu có)
            collection_ref = self.db.collection(collection_name)
            docs = collection_ref.stream()
            
            batch = self.db.batch()
            count = 0
            for doc in docs:
                batch.delete(doc.reference)
                count += 1
                if count % 100 == 0:  # Batch có giới hạn 500 operations
                    batch.commit()
                    batch = self.db.batch()
            
            if count > 0:
                batch.commit()
                print(f"🗑️ Đã xóa {count} documents cũ")
            
            # Upload dữ liệu mới
            batch = self.db.batch()
            for index, row in self.iris_data.iterrows():
                doc_ref = collection_ref.document(f"sample_{index}")
                doc_data = {
                    'sepal_length': float(row['sepal length (cm)']),
                    'sepal_width': float(row['sepal width (cm)']),
                    'petal_length': float(row['petal length (cm)']),
                    'petal_width': float(row['petal width (cm)']),
                    'target': int(row['target']),
                    'target_name': row['target_name'],
                    'sample_id': index,
                    'uploaded_at': datetime.now()
                }
                batch.set(doc_ref, doc_data)
                
                # Commit batch mỗi 100 documents
                if (index + 1) % 100 == 0:
                    batch.commit()
                    batch = self.db.batch()
                    print(f"📤 Đã upload {index + 1} samples...")
            
            # Commit batch cuối cùng
            batch.commit()
            
            print(f"✅ Upload {len(self.iris_data)} samples lên Firebase thành công!")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi upload Firebase: {e}")
            return False
    
    def get_dataset_info(self):
        """Hiển thị thông tin về dataset"""
        if self.iris_data is None:
            print("❌ Chưa có dữ liệu!")
            return
        
        print("\n📊 THÔNG TIN DATASET IRIS:")
        print(f"• Tổng số mẫu: {len(self.iris_data)}")
        print(f"• Số features: {len(self.iris_data.columns) - 2}")  # Trừ target và target_name
        print(f"• Số classes: {self.iris_data['target'].nunique()}")
        print(f"• Classes: {', '.join(self.iris_data['target_name'].unique())}")
        
        print("\n📈 Phân bố classes:")
        class_counts = self.iris_data['target_name'].value_counts()
        for class_name, count in class_counts.items():
            print(f"  - {class_name}: {count} mẫu")
        
        print("\n📋 Mẫu dữ liệu:")
        print(self.iris_data.head())
        
        print("\n📊 Thống kê mô tả:")
        numeric_columns = ['sepal length (cm)', 'sepal width (cm)', 
                          'petal length (cm)', 'petal width (cm)']
        print(self.iris_data[numeric_columns].describe())

def main():
    """Hàm chính để chạy script"""
    print("🌸 IRIS DATASET LOADER")
    print("=" * 50)
    
    # Khởi tạo loader
    loader = IrisDataLoader()
    
    # Tải dữ liệu
    data = loader.load_iris_dataset()
    if data is None:
        return
    
    # Hiển thị thông tin
    loader.get_dataset_info()
    
    # Lưu CSV
    loader.save_to_csv()
    
    # Kết nối và upload Firebase (tùy chọn)
    choice = input("\n🔥 Bạn có muốn upload dữ liệu lên Firebase? (y/n): ").lower()
    if choice == 'y':
        if loader.initialize_firebase():
            loader.upload_to_firebase()
        else:
            print("⚠️ Không thể kết nối Firebase. Vui lòng kiểm tra cấu hình.")
    
    print("\n✅ Hoàn thành!")

if __name__ == "__main__":
    main()