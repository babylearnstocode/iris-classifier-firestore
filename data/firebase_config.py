"""
File: firebase_config.py
Mô tả: Cấu hình và kết nối Firebase
"""

import os
import json
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

# Load environment variables
load_dotenv()

class FirebaseConfig:
    """Class để quản lý cấu hình Firebase"""
    
    def __init__(self):
        self.app = None
        self.db = None
        self.config_path = os.getenv('FIREBASE_CONFIG_PATH', 'firebase-config.json')
        
    def initialize_app(self):
        """Khởi tạo Firebase app"""
        try:
            # Kiểm tra xem app đã được khởi tạo chưa
            if not firebase_admin._apps:
                
                # Cách 1: Sử dụng file JSON
                if os.path.exists(self.config_path):
                    cred = credentials.Certificate(self.config_path)
                    self.app = firebase_admin.initialize_app(cred)
                    print(f"✅ Firebase initialized với file: {self.config_path}")
                
                # Cách 2: Sử dụng environment variables
                else:
                    config_dict = {
                        "type": "service_account",
                        "project_id": os.getenv('FIREBASE_PROJECT_ID'),
                        "private_key_id": os.getenv('FIREBASE_PRIVATE_KEY_ID'),
                        "private_key": os.getenv('FIREBASE_PRIVATE_KEY', '').replace('\\n', '\n'),
                        "client_email": os.getenv('FIREBASE_CLIENT_EMAIL'),
                        "client_id": os.getenv('FIREBASE_CLIENT_ID'),
                        "auth_uri": os.getenv('FIREBASE_AUTH_URI'),
                        "token_uri": os.getenv('FIREBASE_TOKEN_URI'),
                        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                        "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{os.getenv('FIREBASE_CLIENT_EMAIL')}"
                    }
                    
                    # Kiểm tra các biến môi trường cần thiết
                    if config_dict['project_id'] and config_dict['private_key'] and config_dict['client_email']:
                        cred = credentials.Certificate(config_dict)
                        self.app = firebase_admin.initialize_app(cred)
                        print("✅ Firebase initialized với environment variables")
                    else:
                        raise ValueError("Missing required Firebase environment variables")
                
                # Khởi tạo Firestore client
                self.db = firestore.client()
                return True
                
            else:
                # App đã được khởi tạo
                self.app = firebase_admin.get_app()
                self.db = firestore.client()
                print("✅ Firebase app đã được khởi tạo trước đó")
                return True
                
        except Exception as e:
            print(f"❌ Lỗi khởi tạo Firebase: {e}")
            return False
    
    def get_firestore_client(self):
        """Lấy Firestore client"""
        if self.db is None:
            if self.initialize_app():
                return self.db
            else:
                return None
        return self.db
    
    def test_connection(self):
        """Test kết nối Firestore"""
        try:
            if self.db is None:
                if not self.initialize_app():
                    return False
            
            # Tạo một document test
            test_ref = self.db.collection('test').document('connection_test')
            test_ref.set({
                'message': 'Firebase connection test',
                'timestamp': firestore.SERVER_TIMESTAMP
            })
            
            # Đọc lại document
            doc = test_ref.get()
            if doc.exists:
                print("✅ Kết nối Firebase thành công!")
                
                # Xóa document test
                test_ref.delete()
                return True
            else:
                print("❌ Không thể tạo document test")
                return False
                
        except Exception as e:
            print(f"❌ Lỗi test kết nối Firebase: {e}")
            return False
    
    def get_collection_info(self, collection_name):
        """Lấy thông tin về một collection"""
        try:
            if self.db is None:
                if not self.initialize_app():
                    return None
            
            collection_ref = self.db.collection(collection_name)
            docs = collection_ref.stream()
            
            count = 0
            for doc in docs:
                count += 1
                if count > 1000:  # Giới hạn để tránh quá tải
                    break
            
            return {
                'collection_name': collection_name,
                'document_count': count,
                'exists': count > 0
            }
            
        except Exception as e:
            print(f"❌ Lỗi lấy thông tin collection: {e}")
            return None

def create_firebase_config_template():
    """Tạo template file cấu hình Firebase"""
    template = {
        "type": "service_account",
        "project_id": "YOUR_PROJECT_ID",
        "private_key_id": "YOUR_PRIVATE_KEY_ID",
        "private_key": "YOUR_PRIVATE_KEY",
        "client_email": "YOUR_CLIENT_EMAIL",
        "client_id": "YOUR_CLIENT_ID",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/YOUR_CLIENT_EMAIL"
    }
    
    with open('firebase-config-template.json', 'w') as f:
        json.dump(template, f, indent=2)
    
    print("📝 Đã tạo firebase-config-template.json")
    print("💡 Thay thế các giá trị YOUR_* bằng thông tin thực từ Firebase Console")

def main():
    """Test Firebase configuration"""
    print("🔥 FIREBASE CONFIGURATION TEST")
    print("=" * 50)
    
    # Khởi tạo Firebase config
    firebase_config = FirebaseConfig()
    
    # Test kết nối
    if firebase_config.test_connection():
        print("\n📊 Thông tin collections:")
        
        # Kiểm tra collection iris_dataset
        iris_info = firebase_config.get_collection_info('iris_dataset')
        if iris_info:
            print(f"  - iris_dataset: {iris_info['document_count']} documents")
        
        # Kiểm tra collection predictions
        pred_info = firebase_config.get_collection_info('predictions')
        if pred_info:
            print(f"  - predictions: {pred_info['document_count']} documents")
    
    else:
        print("\n💡 Hướng dẫn cấu hình:")
        print("1. Tải file JSON từ Firebase Console")
        print("2. Đặt tên file thành 'firebase-config.json'")
        print("3. Hoặc cấu hình biến môi trường trong .env")
        
        # Tạo template
        create_firebase_config_template()

if __name__ == "__main__":
    main()