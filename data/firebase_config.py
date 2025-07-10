"""
File: firebase_config.py
Mô tả: Cấu hình và kết nối Firebase - Phiên bản cải tiến cho deployment
"""

import os
import json
import logging
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
                
                # Ưu tiên sử dụng environment variables cho deployment
                if self._has_env_variables():
                    logger.info("Attempting to initialize Firebase with environment variables...")
                    if self._init_with_env_vars():
                        return True
                
                # Fallback: Sử dụng file JSON (chủ yếu cho local development)
                elif os.path.exists(self.config_path):
                    logger.info(f"Attempting to initialize Firebase with config file: {self.config_path}")
                    if self._init_with_config_file():
                        return True
                
                # Thử sử dụng Google Cloud credentials mặc định (cho Google Cloud deployment)
                else:
                    logger.info("Attempting to initialize Firebase with default credentials...")
                    if self._init_with_default_credentials():
                        return True
                
                raise ValueError("No valid Firebase configuration found")
                
            else:
                # App đã được khởi tạo
                self.app = firebase_admin.get_app()
                self.db = firestore.client()
                logger.info("Firebase app was already initialized")
                return True
                
        except Exception as e:
            logger.error(f"Firebase initialization failed: {e}")
            return False
    
    def _has_env_variables(self):
        """Kiểm tra xem có đủ environment variables không"""
        required_vars = [
            'FIREBASE_PROJECT_ID',
            'FIREBASE_PRIVATE_KEY',
            'FIREBASE_CLIENT_EMAIL'
        ]
        
        for var in required_vars:
            if not os.getenv(var):
                logger.warning(f"Missing environment variable: {var}")
                return False
        
        return True
    
    def _init_with_env_vars(self):
        """Khởi tạo Firebase với environment variables"""
        try:
            # Lấy private key và xử lý escape characters
            private_key = os.getenv('FIREBASE_PRIVATE_KEY', '')
            
            # Xử lý nhiều format private key khác nhau
            if private_key.startswith('"') and private_key.endswith('"'):
                private_key = private_key[1:-1]  # Bỏ quotes
            
            # Thay thế \n literals thành newline thực
            private_key = private_key.replace('\\n', '\n')
            
            # Đảm bảo private key có format đúng
            if not private_key.startswith('-----BEGIN PRIVATE KEY-----'):
                raise ValueError("Invalid private key format")
            
            config_dict = {
                "type": "service_account",
                "project_id": os.getenv('FIREBASE_PROJECT_ID'),
                "private_key_id": os.getenv('FIREBASE_PRIVATE_KEY_ID'),
                "private_key": private_key,
                "client_email": os.getenv('FIREBASE_CLIENT_EMAIL'),
                "client_id": os.getenv('FIREBASE_CLIENT_ID'),
                "auth_uri": os.getenv('FIREBASE_AUTH_URI', 'https://accounts.google.com/o/oauth2/auth'),
                "token_uri": os.getenv('FIREBASE_TOKEN_URI', 'https://oauth2.googleapis.com/token'),
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{os.getenv('FIREBASE_CLIENT_EMAIL')}"
            }
            
            # Validate config
            if not all([config_dict['project_id'], config_dict['private_key'], config_dict['client_email']]):
                raise ValueError("Missing required Firebase configuration")
            
            cred = credentials.Certificate(config_dict)
            self.app = firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            logger.info("Firebase initialized successfully with environment variables")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize with environment variables: {e}")
            return False
    
    def _init_with_config_file(self):
        """Khởi tạo Firebase với config file"""
        try:
            cred = credentials.Certificate(self.config_path)
            self.app = firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            logger.info(f"Firebase initialized successfully with config file: {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize with config file: {e}")
            return False
    
    def _init_with_default_credentials(self):
        """Khởi tạo Firebase với default credentials (cho Google Cloud)"""
        try:
            # Cho các môi trường như Google Cloud Run, App Engine
            cred = credentials.ApplicationDefault()
            self.app = firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            logger.info("Firebase initialized successfully with default credentials")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize with default credentials: {e}")
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
            
            # Tạo một document test với timeout
            test_ref = self.db.collection('test').document('connection_test')
            test_ref.set({
                'message': 'Firebase connection test',
                'timestamp': firestore.SERVER_TIMESTAMP,
                'environment': os.getenv('ENVIRONMENT', 'unknown')
            })
            
            # Đọc lại document với timeout
            doc = test_ref.get()
            if doc.exists:
                logger.info("Firebase connection test successful!")
                
                # Xóa document test
                test_ref.delete()
                return True
            else:
                logger.error("Failed to create test document")
                return False
                
        except Exception as e:
            logger.error(f"Firebase connection test failed: {e}")
            return False
    
    def get_collection_info(self, collection_name):
        """Lấy thông tin về một collection"""
        try:
            if self.db is None:
                if not self.initialize_app():
                    return None
            
            collection_ref = self.db.collection(collection_name)
            
            # Sử dụng limit để tránh load quá nhiều data
            docs = collection_ref.limit(1000).stream()
            
            count = 0
            for doc in docs:
                count += 1
            
            return {
                'collection_name': collection_name,
                'document_count': count,
                'exists': count > 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None

def create_env_template():
    """Tạo template file .env cho environment variables"""
    template = """# Firebase Configuration
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_PRIVATE_KEY_ID=your-private-key-id
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\nyour-private-key-content\n-----END PRIVATE KEY-----\n"
FIREBASE_CLIENT_EMAIL=your-service-account@your-project-id.iam.gserviceaccount.com
FIREBASE_CLIENT_ID=your-client-id
FIREBASE_AUTH_URI=https://accounts.google.com/o/oauth2/auth
FIREBASE_TOKEN_URI=https://oauth2.googleapis.com/token

# Environment
ENVIRONMENT=production
"""
    
    with open('.env.template', 'w') as f:
        f.write(template)
    
    logger.info("Created .env.template file")
    print("💡 Copy .env.template to .env and replace with your actual Firebase credentials")

def main():
    """Test Firebase configuration"""
    print("🔥 FIREBASE CONFIGURATION TEST")
    print("=" * 50)
    
    # Khởi tạo Firebase config
    firebase_config = FirebaseConfig()
    
    # Test kết nối
    if firebase_config.test_connection():
        print("\n📊 Collection information:")
        
        # Kiểm tra collection iris_dataset
        iris_info = firebase_config.get_collection_info('iris_dataset')
        if iris_info:
            print(f"  - iris_dataset: {iris_info['document_count']} documents")
        
        # Kiểm tra collection predictions
        pred_info = firebase_config.get_collection_info('predictions')
        if pred_info:
            print(f"  - predictions: {pred_info['document_count']} documents")
    
    else:
        print("\n💡 Configuration Guide:")
        print("1. For local development: Use firebase-config.json file")
        print("2. For deployment: Use environment variables")
        print("3. For Google Cloud: Use default credentials")
        
        # Tạo template
        create_env_template()

if __name__ == "__main__":
    main()