import numpy as np

class EnergyTransformer:
    def __init__(self):
        pass

    def _dist_matrix(self, A, B):
        # Tính khoảng cách Euclidean giữa các hàng của A và B
        return np.sqrt(np.sum((A[:, np.newaxis, :] - B[np.newaxis, :, :])**2, axis=2))

    def _calculate_point_to_cluster_energy(self, point, cluster_samples):
        # point: một mẫu đơn lẻ (kích thước: 1 x n_features)
        # cluster_samples: tập hợp các mẫu trong cụm j (kích thước: n_cluster_samples x n_features)
        
        # 1. Thành phần E|x - Y|: Khoảng cách trung bình từ điểm x đến các điểm trong cụm
        # point có 1 dòng nên dist_matrix sẽ trả về ma trận (1 x n_cluster_samples)
        dist_x_y = self._dist_matrix(point, cluster_samples)
        mean_x_y = np.mean(dist_x_y)
        
        # 2. Thành phần E|Y - Y'|: Khoảng cách trung bình giữa các cặp điểm nội bộ trong cụm
        n_cluster_samples = cluster_samples.shape[0]
        if n_cluster_samples > 1:
            dist_y_y = self._dist_matrix(cluster_samples, cluster_samples)
            mean_y_y = np.sum(dist_y_y) / (n_cluster_samples * n_cluster_samples)
        else:
            mean_y_y = 0
            
        # 3. Công thức rút gọn đúng bản chất toán học: Đứng từ mẫu x nhìn sang quần thể cụm Y
        return 2 * mean_x_y - mean_y_y

    def transform(self, X, clusters_population):
        # X: Tập dữ liệu cần ánh xạ (Train hoặc Test) có kích thước (n_samples x n_features)
        # clusters_population: Danh sách chứa dữ liệu thực tế của các cụm mẫu nhận từ DataProcessor
        
        n_samples = X.shape[0]
        n_clusters = len(clusters_population)
        
        # Khởi tạo ma trận năng lượng mới: (Số mẫu ban đầu x Số cụm mẫu)
        X_energy = np.zeros((n_samples, n_clusters))

        # Lặp qua từng mẫu dữ liệu (hàng)
        for i in range(n_samples):
            # Trích xuất mẫu thứ i và đưa về dạng ma trận (1 x n_features)
            current_sample = X[i, :].reshape(1, -1)
            
            # Lặp qua từng cụm quần thể mẫu (cột)
            for j in range(n_clusters):
                cluster_samples = clusters_population[j]
                
                # Tính toán giá trị khoảng cách năng lượng trực tiếp
                X_energy[i, j] = self._calculate_point_to_cluster_energy(current_sample, cluster_samples)
        
        return X_energy