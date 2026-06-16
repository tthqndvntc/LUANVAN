import os
import sys
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Tắt Numba JIT tương tự bản Streamlit để tránh lỗi cache
os.environ['NUMBA_DISABLE_JIT'] = '1'
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'

# Import các module core của bạn
from data_processor import DataProcessor
from energytransformer import EnergyTransformer
from model_trainer import ModelTrainer
from evaluator import Evaluator

def run_energy_svm_pipeline(file_obj, cluster_k, c_param, test_size):
    """
    Hàm xử lý lõi nhận Input từ giao diện Gradio, thực thi thuật toán
    và trả về Output gồm Text kết quả và Hình ảnh biểu đồ.
    """
    if file_obj is None:
        return "❌ Error: Please upload a valid CSV dataset file.", None
        
    # Đọc file dữ liệu từ đối tượng Gradio truyền vào
    try:
        df = pd.read_csv(file_obj.name)
    except Exception as e:
        return f"❌ Error reading CSV file: {str(e)}", None

    processor = DataProcessor()
    transformer = EnergyTransformer()
    
    # 10 hạt giống cố định bảo đảm tính tái lặp quốc tế
    seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    accuracy_list = []
    
    # Lưu ma trận nhầm lẫn của lượt chạy cuối cùng để vẽ biểu đồ minh họa
    last_cm = None 

    # Chạy vòng lặp 10 lần
    for seed in seeds:
        X_train_scaled, X_test_scaled, y_train, y_test, classes, clusters_population = processor.process(
            df, test_size=test_size, n_energy_clusters=int(cluster_k), random_state=seed
        )
        
        X_train_energy = transformer.transform(X_train_scaled, clusters_population)
        X_test_energy = transformer.transform(X_test_scaled, clusters_population)
        
        e_scaler = StandardScaler()
        X_train_energy = e_scaler.fit_transform(X_train_energy)
        X_test_energy = e_scaler.transform(X_test_energy)
        
        trainer = ModelTrainer(C=c_param, kernel='rbf', gamma='scale', random_state=seed)
        trainer.train(X_train_energy, y_train)
        
        y_pred = trainer.predict(X_test_energy)
        acc, cm = Evaluator.get_results(y_test, y_pred)
        accuracy_list.append(acc)
        last_cm = cm

    # Tính toán kết quả thống kê tổng hợp
    mean_accuracy = np.mean(accuracy_list)
    std_accuracy = np.std(accuracy_list)
    
    # Định dạng chuỗi văn bản kết quả hiển thị (Tiếng Anh học thuật)
    result_text = (
        f"============ 📈 AGGREGATED STATISTICAL RESULTS ============\n"
        f"• Number of Prototype Clusters (K): {cluster_k}\n"
        f"• SVM Regularization Parameter (C): {c_param}\n"
        f"• Testing Split Ratio: {test_size}\n"
        f"-----------------------------------------------------------\n"
        f"✅ MEAN CLASSIFICATION ACCURACY: {mean_accuracy * 100:.2f} %\n"
        f"⚠️ STANDARD DEVIATION (Std): {std_accuracy * 100:.2f} %\n"
        f"==========================================================="
    )
    
    # Vẽ biểu đồ Confusion Matrix bằng Matplotlib & Seaborn
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(last_cm, annot=True, fmt='d', cmap="Purples", ax=ax)
    ax.set_title("Confusion Matrix (Last Run)")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    plt.tight_layout()
    
    return result_text, fig

# --- XÂY DỰNG GIAO DIỆN GRADIO (BLOCKS STYLE) ---
with gr.Blocks(title="Energy-SVM Research Pipeline") as demo:
    gr.Markdown("# 🔬 SVM WITH ENERGY-BASED DISTANCE METRIC")
    gr.Markdown("### Prototype Cluster-Based Classification Framework (Gradio Version)")
    
    with gr.Row():
        # Cột bên trái: Các thành phần Input
        with gr.Column():
            gr.Markdown("### 📁 Input Configurations")
            file_input = gr.File(label="Upload CSV Dataset", file_types=[".csv"])
            cluster_slider = gr.Slider(minimum=2, maximum=50, value=10, step=1, label="Number of Representative Clusters (K)")
            c_input = gr.Number(value=10.0, label="SVM Regularization Parameter (C)")
            test_size_slider = gr.Slider(minimum=0.1, maximum=0.5, value=0.3, step=0.05, label="Testing Split Ratio (Test Size)")
            
            submit_btn = gr.Button("🚀 Execute Experimental Evaluation", variant="primary")
            
        # Cột bên phải: Các thành phần Output
        with gr.Column():
            gr.Markdown("### 📊 Experimental Performance")
            txt_output = gr.Textbox(label="Statistical Breakdown Summary", lines=8)
            plot_output = gr.Plot(label="Confusion Matrix Visualization")
            
    # Cấu hình sự kiện click nút bấm
    submit_btn.click(
        fn=run_energy_svm_pipeline,
        inputs=[file_input, cluster_slider, c_input, test_size_slider],
        outputs=[txt_output, plot_output]
    )

if __name__ == "__main__":
    # CHỐT: share=True giúp tự động sinh link công khai mà không cần cài Ngrok
    demo.launch(share=True)
    