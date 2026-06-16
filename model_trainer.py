from sklearn.svm import SVC

class ModelTrainer:
    def __init__(self, C=1.0, kernel='rbf', gamma='scale', random_state=42):
        """
        C: Tham số điều chỉnh (Regularization).
        kernel: Dùng 'rbf' (mặc định) để SVC tự tính toán độ tương đồng 
                trong không gian năng lượng mới.
        gamma: Quyết định tầm ảnh hưởng của một mẫu huấn luyện đơn lẻ.
        """
        # CHỐT: Bỏ hoàn toàn kernel='precomputed'
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=random_state)

    def train(self, X_train_energy, y_train):
        """
        X_train_energy: Tập hợp các điểm năng lượng (N mẫu x M cặp tương tác túi).
        y_train: Nhãn lớp tương ứng.
        """
        # SVC tự động học cách chia cắt không gian năng lượng này
        self.model.fit(X_train_energy, y_train)

    def predict(self, X_test_energy):
        """
        X_test_energy: Các điểm năng lượng của tập kiểm tra.
        """
        return self.model.predict(X_test_energy)