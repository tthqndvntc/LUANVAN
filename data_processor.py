import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

class DataProcessor:
    def __init__(self):
        pass

    def process(self, df, test_size=0.3, n_energy_clusters=10, random_state=42):
        # Làm sạch các hàng trống hoàn toàn
        df = df.dropna(how='all').copy()
        n_rows, n_cols = df.shape

        # =========================================================================
        # RULE 1: TỰ ĐỘNG DÒ TÌM VÀ TRÍCH XUẤT CỘT NHÃN (TỰ ĐỘNG NHẬN DIỆN ĐA LỚP)
        # =========================================================================
        target_col = None
        
        # Ưu tiên 1: Tìm cột cuối hoặc cột đầu có kiểu chữ (Object/String)
        for col_idx in [-1, 0]:
            col_name = df.columns[col_idx]
            if df[col_name].dtype == 'object':
                target_col = col_name
                break
                
        # Ưu tiên 2: Tìm cột nào ở cuối bảng mà có số lượng giá trị phân biệt nhỏ (2 hoặc 3, 4 lớp)
        if target_col is None:
            for col in reversed(df.columns):
                if 2 <= df[col].nunique() <= 5:
                    target_col = col
                    break
                    
        # Mặc định: Nếu không khớp gì thì lấy cột cuối cùng làm nhãn
        if target_col is None:
            target_col = df.columns[-1]

        # Trích xuất nhãn thô và mã hóa sạch sẽ (Hỗ trợ chuẩn từ 2 lớp đến đa lớp)
        y_raw = df[target_col].astype(str).str.strip()
        le_y = LabelEncoder()
        y = le_y.fit_transform(y_raw)
        classes = le_y.classes_
        unique_classes = np.unique(y)
        n_classes = len(unique_classes) # Số lượng lớp thực tế (Ví dụ: Iris/Tae = 3, Pima = 2)

        # Loại bỏ cột nhãn ra khỏi ma trận thuộc tính X
        X_raw = df.drop(columns=[target_col]).copy()

        # =========================================================================
        # RULE 2: TỰ ĐỘNG LỌC BỎ CỘT ID VÀ CỘT HẰNG SỐ (ANOMALY FILTERING)
        # =========================================================================
        valid_cols = []
        for col in X_raw.columns:
            n_unique = X_raw[col].nunique()
            if n_unique <= 1:
                continue
            if 'id' in str(col).lower() or n_unique == n_rows:
                continue
            valid_cols.append(col)
            
        X_filtered = X_raw[valid_cols].copy()

        # =========================================================================
        # RULE 3: TỰ ĐỘNG PHÁT HIỆN SỐ 0 GIẢ LẬP KHUYẾT DỮ LIỆU
        # =========================================================================
        for col in X_filtered.columns:
            if X_filtered[col].dtype != 'object':
                zero_ratio = (X_filtered[col] == 0).sum() / n_rows
                if 0.15 < zero_ratio < 0.80 and X_filtered[col].max() > 10:
                    if 'pregnan' not in str(col).lower():
                        X_filtered[col] = X_filtered[col].replace(0, np.nan)

        # =========================================================================
        # RULE 4: PHÂN LOẠI THUỘC TÍNH (CONTINUOUS VS CATEGORICAL)
        # =========================================================================
        continuous_cols = []
        categorical_cols = []

        for col in X_filtered.columns:
            if X_filtered[col].dtype == 'object':
                categorical_cols.append(col)
            else:
                if X_filtered[col].nunique() < 10:
                    categorical_cols.append(col)
                else:
                    continuous_cols.append(col)

        for col in categorical_cols:
            le_x = LabelEncoder()
            X_filtered[col] = le_x.fit_transform(X_filtered[col].astype(str))

        # =========================================================================
        # LUỒNG XỬ LÝ TRAIN/TEST KHÁCH QUAN KHÔNG RÒ RỈ DỮ LIỆU
        # =========================================================================
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_filtered, y, 
            test_size=test_size, 
            shuffle=True,
            stratify=y, # Giữ nguyên tỉ lệ phân bố đa lớp chuẩn xác ở cả 2 tập
            random_state=random_state
        )

        train_median = X_train_raw.median()
        X_train_raw = X_train_raw.fillna(train_median).fillna(0)
        X_test_raw = X_test_raw.fillna(train_median).fillna(0)

        if len(continuous_cols) > 0:
            scaler = StandardScaler()
            X_train_cont_scaled = scaler.fit_transform(X_train_raw[continuous_cols])
            X_train_scaled = np.hstack((X_train_cont_scaled, X_train_raw[categorical_cols].values)) if len(categorical_cols) > 0 else X_train_cont_scaled
            
            X_test_cont_scaled = scaler.transform(X_test_raw[continuous_cols])
            X_test_scaled = np.hstack((X_test_cont_scaled, X_test_raw[categorical_cols].values)) if len(categorical_cols) > 0 else X_test_cont_scaled
        else:
            X_train_scaled = X_train_raw.values
            X_test_scaled = X_test_raw.values

        # =========================================================================
        # ALGORITHM 5: PHÂN CỤM MẪU ĐỘC LẬP THEO TỪNG LỚP ĐA THÍCH NGHI (MỚI)
        # =========================================================================
        clusters_population = []
        
        # Tự động chia đều tổng số cụm K cho tổng số lớp xuất hiện trong file
        # Ví dụ: K=9, có 3 lớp (Iris) -> Mỗi lớp nhận đúng k_sub = 3 cụm đại diện thuần chủng
        clusters_per_class = np.array_split(np.arange(n_energy_clusters), n_classes)
        
        # Vòng lặp tự động quét qua từng lớp độc lập (Bất kể bao nhiêu lớp)
        for c_idx, class_val in enumerate(unique_classes):
            k_c = len(clusters_per_class[c_idx]) # Số cụm chỉ định cho lớp này
            
            # Lọc lấy các mẫu thuộc lớp hiện tại trên tập Train
            X_c = X_train_scaled[y_train == class_val]
            k_c = min(k_c, len(X_c)) # Đảm bảo số cụm không vượt quá số mẫu thực tế
            
            if k_c > 0:
                kmeans = KMeans(n_clusters=k_c, init='k-means++', n_init=20, random_state=random_state)
                kmeans.fit(X_c)
                for i in range(k_c):
                    clusters_population.append(X_c[kmeans.labels_ == i])
                
        return X_train_scaled, X_test_scaled, y_train, y_test, classes, clusters_population