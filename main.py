import os
os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable Numba JIT để tránh lỗi cache
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler 
from data_processor import DataProcessor
from energytransformer import EnergyTransformer
from model_trainer import ModelTrainer
from evaluator import Evaluator 

st.set_page_config(page_title="Energy-SVM Research", page_icon="🔬", layout="wide")
st.title("🔬 SVM WITH ENERGY-BASED DISTANCE")
st.markdown("---")

# --- SIDEBAR: QUẢN LÝ DỮ LIỆU ---
st.sidebar.header("📁 Data Management")
data_folder = "data"
if not os.path.exists(data_folder): os.makedirs(data_folder)

list_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
option = st.sidebar.selectbox("Data Source", ["Select file in data folder", "Upload new file"])

df = None
if option == "Select file in data folder" and list_files:
    selected_file = st.sidebar.selectbox("💡Experimental Datasets", list_files)
    df = pd.read_csv(os.path.join(data_folder, selected_file))
elif option == "Upload new file":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file: df = pd.read_csv(uploaded_file)
        
# Tham số mô hình
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Model Configuration")
C_param = st.sidebar.number_input("SVM Regularization Parameter (C)", 0.001, 500.0, 10.0)
n_energy_clusters = st.sidebar.number_input("Number of Representative Clusters (K)", min_value=2, max_value=50, value=10, step=1)
test_size_ratio = st.sidebar.slider("Testing Split Ratio (Test Size)", 0.1, 0.5, 0.3)

if df is not None:
    # =========================================================================
    # 🌟 DATASET PREVIEW & METRICS (CLOUD DEPLOYMENT OPTIMIZED - HARDCODED BENCHMARKS)
    # =========================================================================
    st.markdown("### 🔍 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    # =========================================================================

    # 1. KHỜI TẠO XỬ LÝ (Nhận danh sách các cụm quần thể mẫu thay vì groups cột)
    processor = DataProcessor()
    X_train_scaled, X_test_scaled, y_train, y_test, classes, clusters_population = processor.process(
        df, test_size=test_size_ratio, n_energy_clusters=n_energy_clusters
    )
    
    st.info(f"✅ Data loaded successfully.")
    st.success(f"✅ Training set successfully partitioned into {len(clusters_population)} representative sample clusters.")

    # Khởi tạo không gian lưu trữ trạng thái trên RAM nếu chưa có
    if 'pipeline_executed' not in st.session_state:
        st.session_state.pipeline_executed = False

    # Bắt sự kiện bấm nút sidebar để khóa cờ trạng thái
    if st.sidebar.button("🚀 Execute Experimental Evaluation"):
        st.session_state.pipeline_executed = True

    # NẾU CỜ TRẠNG THÁI LÀ TRUE -> ĐÓNG BĂNG GIAO DIỆN HIỂN THỊ KỂ CẢ KHI BẤM TẢI FILE
    if st.session_state.pipeline_executed:
        transformer = EnergyTransformer()
        
        # 10 hạt giống cố định bảo đảm tính tái lặp quốc tế
        seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        accuracy_list = []
        
        # Trích xuất chính xác tên các cột đặc trưng sạch cho việc xuất file định dạng cột
        temp_df = df.dropna(how='all').copy()
        target_col = None
        for col_idx in [-1, 0]:
            col_name = temp_df.columns[col_idx]
            if temp_df[col_name].dtype == 'object':
                target_col = col_name
                break
        if target_col is None:
            for col in reversed(temp_df.columns):
                if 2 <= temp_df[col].nunique() <= 5:
                    target_col = col
                    break
        if target_col is None:
            target_col = temp_df.columns[-1]
            
        features_preview = [c for c in temp_df.columns if c != target_col and not c.lower().startswith('unnamed')]
        
        # Đồng bộ hóa hạt giống đầu tiên (Run 1) cố định với seed=42 để lấy ma trận làm tài nguyên xuất file
        X_tr_s_file, X_te_s_file, y_tr_file, y_te_file, classes_file, clusters_file = processor.process(
            df, test_size=test_size_ratio, n_energy_clusters=n_energy_clusters, random_state=42
        )
        
        # Giải mã ngược nhãn chuỗi nếu hệ thống sử dụng LabelEncoder
        le = processor.le if hasattr(processor, 'le') else None
        y_train_orig = le.inverse_transform(y_tr_file) if le is not None else y_tr_file
        y_test_orig = le.inverse_transform(y_te_file) if le is not None else y_te_file

        # Tính ma trận năng lượng chuẩn hóa mẫu để phục vụ kết xuất file chặng 2
        X_tr_energy_sample = transformer.transform(X_tr_s_file, clusters_file)
        X_te_energy_sample = transformer.transform(X_te_s_file, clusters_file)
        e_scaler_sample = StandardScaler()
        X_tr_energy_sample = e_scaler_sample.fit_transform(X_tr_energy_sample)
        X_te_energy_sample = e_scaler_sample.transform(X_te_energy_sample)

        # =========================================================================
        # 🌟 EXPORT BLOCK 1: CLEAN ATTRIBUTE SPACE (3 FILES)
        # =========================================================================
        st.markdown("### 📥 1. Clean Attribute Space Partitions (Pre-Transformation)")
        
        df_train_file = pd.DataFrame(X_tr_s_file)
        df_train_file.columns = features_preview if df_train_file.shape[1] == len(features_preview) else [f"Feature_{i+1}" for i in range(df_train_file.shape[1])]
        df_train_file = df_train_file.reset_index(drop=True)
        df_train_file['Class_Label'] = list(y_train_orig)
        
        df_test_file = pd.DataFrame(X_te_s_file)
        df_test_file.columns = features_preview if df_test_file.shape[1] == len(features_preview) else [f"Feature_{i+1}" for i in range(df_test_file.shape[1])]
        df_test_file = df_test_file.reset_index(drop=True)
        
        df_test_label_file = pd.DataFrame({'True_Label': list(y_test_orig)})

        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("📥 Download train.csv", data=df_train_file.to_csv(index=False).encode('utf-8'), file_name="train.csv", mime="text/csv", use_container_width=True)
        with c2:
            st.download_button("📤 Download test.csv (🔒 Hidden Labels)", data=df_test_file.to_csv(index=False).encode('utf-8'), file_name="test.csv", mime="text/csv", use_container_width=True)
        with c3:
            st.download_button("🔑 Download test_label.csv (Answers)", data=df_test_label_file.to_csv(index=False).encode('utf-8'), file_name="test_label.csv", mime="text/csv", use_container_width=True)

        # =========================================================================
        # 🌟 EXPORT BLOCK 2: EBD ENERGY SPACE MATRIX (2 FILES)
        # =========================================================================
        st.markdown("### 🧬 2. Standardized Energy Space Matrices (Post-Transformation)")
        
        cluster_names = [f"Energy_To_Cluster_{i+1}" for i in range(X_tr_energy_sample.shape[1])]
        df_train_energy_file = pd.DataFrame(X_tr_energy_sample, columns=cluster_names)
        df_test_energy_file = pd.DataFrame(X_te_energy_sample, columns=cluster_names)
        
        ce1, ce2 = st.columns(2)
        with ce1:
            st.download_button("⚡ Download train_energy.csv", data=df_train_energy_file.to_csv(index=False).encode('utf-8'), file_name="train_energy.csv", mime="text/csv", use_container_width=True)
        with ce2:
            st.download_button("⚡ Download test_energy.csv", data=df_test_energy_file.to_csv(index=False).encode('utf-8'), file_name="test_energy.csv", mime="text/csv", use_container_width=True)
        st.markdown("---")
        # =========================================================================

        st.markdown("### 📊 EXPERIMENTAL RESULTS OVER 10 INDEPENDENT RUNS")
        
        # Chạy vòng lặp 10 lần
        for run, seed in enumerate(seeds):
            X_train_scaled, X_test_scaled, y_train, y_test, classes, clusters_population = processor.process(
                df, test_size=test_size_ratio, n_energy_clusters=n_energy_clusters, random_state=seed
            )
            
            X_train_energy = transformer.transform(X_train_scaled, clusters_population)
            X_test_energy = transformer.transform(X_test_scaled, clusters_population)
            
            e_scaler = StandardScaler()
            X_train_energy = e_scaler.fit_transform(X_train_energy)
            X_test_energy = e_scaler.transform(X_test_energy)
            
            trainer = ModelTrainer(C=C_param, kernel='rbf', gamma='scale', random_state=seed)
            trainer.train(X_train_energy, y_train)
            
            y_pred = trainer.predict(X_test_energy)
            acc, cm = Evaluator.get_results(y_test, y_pred)
            accuracy_list.append(acc)
            
            # --- ĐƯA HIỂN THỊ CHI TIẾT VÀO TRONG VÒNG LẶP DÙNG EXPANDER ---
            with st.expander(f"🔹 Run Number {run + 1} ——— Accuracy: {acc*100:.2f} %"):
                col1, col2 = st.columns([1, 1.2])
                
                with col1:
                    st.metric("Accuracy", f"{acc*100:.2f} %")
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap="Purples", ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("True")
                    st.pyplot(fig)
                    plt.close(fig) # Đóng fig để giải phóng bộ nhớ hệ thống
                    
                with col2:
                    st.markdown("#### Detailed Classification Breakdown")
                    res_df = pd.DataFrame({
                        'Sample ID': range(len(y_test)),
                        'Predicted Class': [classes[i] for i in y_pred],
                        'True Class': [classes[i] for i in y_test]
                        
                    })
                    res_df['Evaluation'] = ["✅ CORRECT" if a == b else "❌ INCORRECT" for a, b in zip(res_df['True Class'], res_df['Predicted Class'])]
                    st.dataframe(res_df, use_container_width=True)
                    st.info(f"Total test samples: {len(y_test)} | Correctly predicted samples: {sum(y_test == y_pred)}")

        # --- TÍNH TOÁN VÀ HIỂN THỊ KẾT QUẢ THỐNG KÊ TỔNG HỢP ---
        mean_accuracy = np.mean(accuracy_list)
        std_accuracy = np.std(accuracy_list)
        
        st.markdown("---")
        st.markdown("#### 📈 AGGREGATED FINAL STATISTICAL BREAKDOWN")
        
        col_mean, col_std = st.columns(2)
        with col_mean:
            st.metric(label="Mean Classification Accuracy", value=f"{mean_accuracy * 100:.2f} %")
        with col_std:
            st.metric(label="Standard Deviation (Std)", value=f"{std_accuracy * 100:.2f} %")