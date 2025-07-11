"""
File: visualize_iris.py
Mô tả: Trực quan hóa dữ liệu Iris dataset với 3 biểu đồ cơ bản
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class IrisVisualizer:
    def __init__(self, csv_path="data/processed/iris_dataset.csv"):
        """Khởi tạo IrisVisualizer"""
        self.csv_path = csv_path
        self.data = None
        self.feature_columns = ['sepal length (cm)', 'sepal width (cm)', 
                               'petal length (cm)', 'petal width (cm)']
        
        # Thiết lập style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_data(self):
        """Tải dữ liệu từ file CSV"""
        try:
            if not Path(self.csv_path).exists():
                print(f"❌ File không tồn tại: {self.csv_path}")
                return False
            
            self.data = pd.read_csv(self.csv_path)
            print(f"✅ Tải dữ liệu thành công: {len(self.data)} mẫu")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi tải dữ liệu: {e}")
            return False
    
    def plot_histogram(self):
        """1. Biểu đồ histogram cho các features"""
        if self.data is None:
            print("❌ Chưa tải dữ liệu!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle('Biểu đồ phân phối các đặc trưng', fontsize=14, fontweight='bold')
        
        for i, feature in enumerate(self.feature_columns):
            row = i // 2
            col = i % 2
            
            axes[row, col].hist(self.data[feature], bins=15, alpha=0.7, 
                               color='skyblue', edgecolor='black')
            axes[row, col].set_title(feature, fontweight='bold')
            axes[row, col].set_xlabel('Giá trị (cm)')
            axes[row, col].set_ylabel('Tần suất')
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_boxplot(self):
        """2. Boxplot theo từng loài hoa"""
        if self.data is None:
            print("❌ Chưa tải dữ liệu!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle('Boxplot các đặc trưng theo loài hoa', fontsize=14, fontweight='bold')
        
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        
        for i, feature in enumerate(self.feature_columns):
            row = i // 2
            col = i % 2
            
            sns.boxplot(data=self.data, x='target_name', y=feature, 
                       ax=axes[row, col], palette=colors)
            axes[row, col].set_title(feature, fontweight='bold')
            axes[row, col].set_xlabel('Loài hoa')
            axes[row, col].set_ylabel('Giá trị (cm)')
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_scatter(self):
        """3. Scatter plot giữa 2 đặc trưng quan trọng nhất"""
        if self.data is None:
            print("❌ Chưa tải dữ liệu!")
            return
        
        # Tạo 2 subplot: petal length vs petal width và sepal length vs sepal width
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Scatter Plot các đặc trưng chính', fontsize=14, fontweight='bold')
        
        colors = {'setosa': 'red', 'versicolor': 'blue', 'virginica': 'green'}
        
        # Plot 1: Petal length vs Petal width
        for species in self.data['target_name'].unique():
            data_subset = self.data[self.data['target_name'] == species]
            axes[0].scatter(data_subset['petal length (cm)'], 
                           data_subset['petal width (cm)'], 
                           c=colors[species], label=species, alpha=0.7, s=50)
        
        axes[0].set_xlabel('Petal Length (cm)')
        axes[0].set_ylabel('Petal Width (cm)')
        axes[0].set_title('Petal Length vs Petal Width')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Sepal length vs Sepal width
        for species in self.data['target_name'].unique():
            data_subset = self.data[self.data['target_name'] == species]
            axes[1].scatter(data_subset['sepal length (cm)'], 
                           data_subset['sepal width (cm)'], 
                           c=colors[species], label=species, alpha=0.7, s=50)
        
        axes[1].set_xlabel('Sepal Length (cm)')
        axes[1].set_ylabel('Sepal Width (cm)')
        axes[1].set_title('Sepal Length vs Sepal Width')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def show_basic_info(self):
        """Hiển thị thông tin cơ bản"""
        if self.data is None:
            print("❌ Chưa tải dữ liệu!")
            return
        
        print(f"\n📊 Thông tin cơ bản:")
        print(f"• Tổng số mẫu: {len(self.data)}")
        print(f"• Số đặc trưng: {len(self.feature_columns)}")
        print(f"• Các loài: {', '.join(self.data['target_name'].unique())}")
        
        print(f"\n📈 Phân bố theo loài:")
        for species, count in self.data['target_name'].value_counts().items():
            print(f"  - {species}: {count} mẫu")
    
    def visualize_all(self):
        """Hiển thị tất cả 3 biểu đồ cơ bản"""
        print("🎨 TRỰC QUAN HÓA DỮ LIỆU IRIS")
        print("=" * 40)
        
        if not self.load_data():
            return
        
        self.show_basic_info()
        
        print("\n📊 Đang tạo các biểu đồ...")
        
        print("1. Histogram...")
        self.plot_histogram()
        
        print("2. Boxplot...")
        self.plot_boxplot()
        
        print("3. Scatter Plot...")
        self.plot_scatter()
        
        print("\n✅ Hoàn thành!")

def main():
    """Hàm chính"""
    print("🌸 IRIS DATASET VISUALIZER")
    print("=" * 40)
    
    # Khởi tạo và chạy visualizer
    visualizer = IrisVisualizer()
    visualizer.visualize_all()
    
if __name__ == "__main__":
    main()