import pandas as pd
import os
import streamlit as st
from sklearn.datasets import load_iris
from firebase_admin import firestore
import logging

class DataLoader:
    def __init__(self):
        self.local_dataset_path = 'data/iris_dataset.csv'
        self.firestore_collection = 'iris_dataset'
        
    def check_local_dataset_exists(self):
        """
        Kiểm tra xem file iris_dataset.csv có tồn tại trong local không
        
        Returns:
            bool: True nếu file tồn tại, False nếu không
        """
        try:
            return os.path.exists(self.local_dataset_path)
        except Exception as e:
            st.error(f"Error checking local dataset: {str(e)}")
            return False
    
    def load_local_dataset(self):
        """
        Load dataset từ file CSV local
        
        Returns:
            pandas.DataFrame or None: DataFrame nếu load thành công, None nếu thất bại
        """
        try:
            if not self.check_local_dataset_exists():
                return None
                
            df = pd.read_csv(self.local_dataset_path)
            
            # Validate dataset structure
            required_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
            if all(col in df.columns for col in required_columns):
                st.info(f"📁 Dataset loaded from local file: {self.local_dataset_path}")
                return df
            else:
                st.warning(f"Local dataset found but missing required columns: {required_columns}")
                return None
                
        except Exception as e:
            st.error(f"Error loading local dataset: {str(e)}")
            return None
    
    def load_from_firestore(self, db):
        """
        Load dataset từ Firestore collection
        
        Args:
            db: Firestore database client
            
        Returns:
            pandas.DataFrame or None: DataFrame nếu load thành công, None nếu thất bại
        """
        try:
            if not db:
                st.warning("Firestore client not available")
                return None
                
            # Lấy tất cả documents từ collection
            docs = db.collection(self.firestore_collection).stream()
            
            # Chuyển đổi documents thành list of dictionaries
            data = []
            for doc in docs:
                doc_data = doc.to_dict()
                data.append(doc_data)
            
            if not data:
                st.info("No data found in Firestore collection")
                return None
                
            # Chuyển đổi thành DataFrame
            df = pd.DataFrame(data)
            
            # Validate dataset structure
            required_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
            if all(col in df.columns for col in required_columns):
                st.success(f"☁️ Dataset loaded from Firestore ({len(df)} records)")
                return df
            else:
                st.warning(f"Firestore dataset found but missing required columns: {required_columns}")
                return None
                
        except Exception as e:
            st.error(f"Error loading dataset from Firestore: {str(e)}")
            return None
    
    def save_to_firestore(self, df, db):
        """
        Save dataset lên Firestore collection
        
        Args:
            df: pandas.DataFrame - Dataset cần save
            db: Firestore database client
            
        Returns:
            bool: True nếu save thành công, False nếu thất bại
        """
        try:
            if not db:
                st.warning("Firestore client not available")
                return False
                
            # Chuyển DataFrame thành list of dictionaries
            records = df.to_dict('records')
            
            # Xóa collection cũ trước khi save mới (optional)
            batch = db.batch()
            
            # Lấy tất cả documents hiện có để xóa
            existing_docs = db.collection(self.firestore_collection).stream()
            for doc in existing_docs:
                batch.delete(doc.reference)
            
            # Commit batch delete
            batch.commit()
            
            # Tạo batch mới để add data
            batch = db.batch()
            
            # Add từng record vào Firestore
            for record in records:
                doc_ref = db.collection(self.firestore_collection).document()
                batch.set(doc_ref, record)
            
            # Commit batch add
            batch.commit()
            
            st.success(f"☁️ Dataset saved to Firestore ({len(records)} records)")
            return True
            
        except Exception as e:
            st.error(f"Error saving dataset to Firestore: {str(e)}")
            return False
    
    def save_to_local(self, df):
        """
        Save dataset xuống file CSV local
        
        Args:
            df: pandas.DataFrame - Dataset cần save
            
        Returns:
            bool: True nếu save thành công, False nếu thất bại
        """
        try:
            # Tạo thư mục data nếu chưa tồn tại
            os.makedirs("data", exist_ok=True)
            
            # Save DataFrame thành CSV
            df.to_csv(self.local_dataset_path, index=False)
            
            st.success(f"💾 Dataset saved to local file: {self.local_dataset_path}")
            return True
            
        except Exception as e:
            st.error(f"Error saving dataset to local: {str(e)}")
            return False
    
    def load_sklearn_iris(self):
        """
        Load iris dataset từ sklearn
        
        Returns:
            pandas.DataFrame or None: DataFrame nếu load thành công, None nếu thất bại
        """
        try:
            iris = load_iris()
            df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
            df['species'] = iris.target_names[iris.target]
            
            st.success(f"📊 Fresh Iris dataset loaded from sklearn ({len(df)} records)")
            return df
            
        except Exception as e:
            st.error(f"Error loading iris dataset from sklearn: {str(e)}")
            return None
    
    def initialize_dataset(self, db=None):
        """
        Initialize dataset theo logic ưu tiên mới:
        1. Kiểm tra local dataset trước
        2. Nếu không có local và có Firestore connection -> bắt buộc load từ Firestore
        3. Nếu không có cả hai -> hiển thị thông báo cần get data từ Firestore
        
        Args:
            db: Firestore database client (optional)
            
        Returns:
            pandas.DataFrame or None: Dataset đã load, None nếu thất bại hoặc cần get data
        """
        try:
            # Bước 1: Kiểm tra local dataset trước
            if self.check_local_dataset_exists():
                df = self.load_local_dataset()
                if df is not None:
                    return df
            
            # Bước 2: Nếu không có local, KHÔNG tự động tải từ Firestore nữa
            st.warning("⚠️ No local dataset found.")
            st.info("💡 Please use the 'Load from Firestore' button below to load data manually.")
            return None
            
        except Exception as e:
            st.error(f"Error initializing dataset: {str(e)}")
            return None
    
    def get_data_from_firestore_button(self, db):
        """
        Hiển thị button để get data từ Firestore và xử lý logic
        
        Args:
            db: Firestore database client
            
        Returns:
            pandas.DataFrame or None: Dataset nếu load thành công, None nếu thất bại
        """
        try:
            if not db:
                st.error("Firestore client not available. Please check your connection.")
                return None
            
            # Hiển thị button
            if st.button("🔄 Get Data from Firestore", type="primary"):
                with st.spinner("Loading data from Firestore..."):
                    df = self.load_from_firestore(db)
                    if df is not None:
                        # Save xuống local
                        self.save_to_local(df)
                        st.success("✅ Data successfully loaded from Firestore and saved to local!")
                        # Rerun để refresh page
                        st.rerun()
                        return df
                    else:
                        st.error("❌ No data found in Firestore collection.")
                        return None
            
            return None
            
        except Exception as e:
            st.error(f"Error getting data from Firestore: {str(e)}")
            return None
    
    def check_data_availability(self, db=None):
        """
        Kiểm tra tính khả dụng của dữ liệu và hiển thị UI tương ứng
        
        Args:
            db: Firestore database client (optional)
            
        Returns:
            pandas.DataFrame or None: Dataset nếu có sẵn, None nếu cần get data
        """
        try:
            # Kiểm tra local trước
            if self.check_local_dataset_exists():
                df = self.load_local_dataset()
                if df is not None:
                    return df
            
            # Không có local data
            st.warning("⚠️ No local dataset found.")
            
            if db is not None:
                # Có Firestore connection, hiển thị button get data
                st.info("💡 You can get data from Firestore cloud storage.")
                return self.get_data_from_firestore_button(db)
            else:
                # Không có Firestore connection
                st.error("❌ No Firestore connection available. Please check your Firebase configuration.")
                return None
                
        except Exception as e:
            st.error(f"Error checking data availability: {str(e)}")
            return None
    
    def get_dataset_info(self, df):
        """
        Lấy thông tin cơ bản về dataset
        
        Args:
            df: pandas.DataFrame - Dataset
            
        Returns:
            dict: Dictionary chứa thông tin về dataset
        """
        try:
            if df is None:
                return None
                
            info = {
                'total_records': len(df),
                'features': len(df.columns) - 1,  # Exclude target column
                'species_count': len(df['species'].unique()),
                'species_list': df['species'].unique().tolist(),
                'species_distribution': df['species'].value_counts().to_dict(),
                'columns': df.columns.tolist()
            }
            
            return info
            
        except Exception as e:
            st.error(f"Error getting dataset info: {str(e)}")
            return None
    
    def refresh_from_firestore(self, db):
        """
        Refresh dataset từ Firestore (force reload)
        
        Args:
            db: Firestore database client
            
        Returns:
            pandas.DataFrame or None: Dataset mới từ Firestore
        """
        try:
            if not db:
                st.warning("Firestore client not available")
                return None
                
            df = self.load_from_firestore(db)
            if df is not None:
                # Save xuống local để thay thế file cũ
                self.save_to_local(df)
            
            return df
            
        except Exception as e:
            st.error(f"Error refreshing dataset from Firestore: {str(e)}")
            return None
    
    def upload_local_to_firestore(self, db):
        """
        Upload dataset từ local lên Firestore
        
        Args:
            db: Firestore database client
            
        Returns:
            bool: True nếu upload thành công, False nếu thất bại
        """
        try:
            if not db:
                st.warning("Firestore client not available")
                return False
                
            df = self.load_local_dataset()
            if df is not None:
                return self.save_to_firestore(df, db)
            else:
                st.warning("No local dataset found to upload")
                return False
                
        except Exception as e:
            st.error(f"Error uploading local dataset to Firestore: {str(e)}")
            return False