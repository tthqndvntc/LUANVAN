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

st.set_page_config(page_title="Energy-SVM Research Pipeline", page_icon="🔬", layout="wide")
st.title("🔬 SVM WITH ENERGY-BASED DISTANCE ")
st.markdown("---")

# --- SIDEBAR: QUẢN LÝ DỮ LIỆU ---
st.sidebar.header("📁 Data Management")
data_folder = "data"
if not os.path.exists(data_folder): os.makedirs(data_folder)

list_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
option = st.sidebar.selectbox("Data Source", ["Select file from data folder", "Upload new file"])

df = None
if option == "Select file from data folder" and list_files:
    selected_file = st.sidebar.selectbox("Available Datasets", list_files)
    df = pd.read_csv(os.path.join(data_folder, selected_file))
elif option == "Upload new file":
    uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded_file: df = pd.read_csv(uploaded_file)
        
# Tham số mô hình
C_param = st.sidebar.number_input("SVM Regularization Parameter (C)", 0.001, 500.0, 10.0)
test_size_ratio = st.sidebar.slider("Testing Split Ratio (Test Size)", 0.1, 0.5, 0.3)

# --- SIDEBAR: CẤU HÌNH QUẦN THỂ NĂNG LƯỢNG (MỚI THEO Ý THẦY) ---
st.sidebar.markdown("---")
st.sidebar.header("🧬 Prototype Configuration")
# Đổi tên slider từ nhóm đặc trưng sang số lượng cụm mẫu đại diện (số cột năng lượng mới)
n_energy_clusters = st.sidebar.slider("Number of Representative Clusters (K)", 2, 50, 10)

if df is not None:
    # 1. KHỞI TẠO XỬ LÝ (Nhận danh sách các cụm quần thể mẫu thay vì groups cột)
    processor = DataProcessor()
    X_train_scaled, X_test_scaled, y_train, y_test, classes, clusters_population = processor.process(
        df, test_size=test_size_ratio, n_energy_clusters=n_energy_clusters
    )
    
    st.info(f"✅ Dataset loaded successfully.")
    st.success(f"✅ Training set successfully clustered into {len(clusters_population)} representative prototype clusters.")
    st.info(f"Training Sample Size: {len(X_train_scaled)} | Testing Sample Size: {len(X_test_scaled)}")

    if st.sidebar.button("🚀 Execute Experimental Evaluation"):
        transformer = EnergyTransformer()
        
        # 10 hạt giống cố định bảo đảm tính tái lặp quốc tế
        seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        accuracy_list = []
        
        st.markdown("### 📊 EXPERIMENTAL EVALUATION LOGS (OVER 10 INDEPENDENT RUNS)")
        
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
            with st.expander(f"🔹 Run Number {run + 1:02d} ——— Classification Accuracy: {acc*100:.2f} %"):
                col1, col2 = st.columns([1, 1.2])
                
                with col1:
                    st.metric("Classification Accuracy", f"{acc*100:.2f} %")
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap="Purples", ax=ax)
                    ax.set_xlabel("Predicted Labels")
                    ax.set_ylabel("True Labels")
                    st.pyplot(fig)
                    plt.close(fig) # Đóng fig để giải phóng bộ nhớ hệ thống
                    
                with col2:
                    st.markdown("#### Detailed Prediction Analysis")
                    res_df = pd.DataFrame({
                        'Sample ID': range(len(y_test)),
                        'True Class': [classes[i] for i in y_test],
                        'Predicted Class': [classes[i] for i in y_pred]
                    })
                    res_df['Evaluation'] = ["✅ CORRECT" if a == b else "❌ INCORRECT" for a, b in zip(res_df['True Class'], res_df['Predicted Class'])]
                    st.dataframe(res_df, use_container_width=True)
                    st.info(f"Total Test Samples: {len(y_test)} | Correct Predictions: {sum(y_test == y_pred)}")

        # --- TÍNH TOÁN VÀ HIỂN THỊ KẾT QUẢ THỐNG KÊ TỔNG HỢP ---
        mean_accuracy = np.mean(accuracy_list)
        std_accuracy = np.std(accuracy_list)
        
        st.markdown("---")
        st.markdown("#### 📈 AGGREGATED FINAL STATISTICAL BREAKDOWN")
        
        col_mean, col_std = st.columns(2)
        with col_mean:
            st.metric(label="Mean Classification Accuracy (Average)", value=f"{mean_accuracy * 100:.2f} %")
        with col_std:
            st.metric(label="Standard Deviation (Std)", value=f"{std_accuracy * 100:.2f} %")