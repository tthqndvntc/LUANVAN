import os
import sys
import click
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Disable Numba JIT mapping to avoid cache issues
os.environ['NUMBA_DISABLE_JIT'] = '1'
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'

# Import core modules
from data_processor import DataProcessor
from energytransformer import EnergyTransformer
from model_trainer import ModelTrainer
from evaluator import Evaluator

@click.command()
@click.option('--data', type=click.Path(exists=True), required=True, help='Path to the dataset CSV file (e.g., data/iris.csv).')
@click.option('--cluster', default=10, type=int, help='Number of representative prototype clusters (K). Default: 10.')
@click.option('--c_param', default=10.0, type=float, help='SVM regularization parameter (C). Default: 10.0.')
@click.option('--test_size', default=0.3, type=float, help='The testing dataset split ratio. Default: 0.3.')
def main(data, cluster, c_param, test_size):
    """
    🔬 SVM WITH ENERGY-BASED DISTANCE METRIC (CLI VERSION)
    Experiment pipeline: Prototype cluster-based classification over 10 independent runs.
    """
    click.secho("\n====== 🔬 INITIALIZING ENERGY-SVM RESEARCH PIPELINE ======", fg='cyan', bold=True)
    
    # 1. Load Dataset
    try:
        df = pd.read_csv(data)
        click.secho(f"✅ Successfully loaded dataset: {data}", fg='green')
    except Exception as e:
        click.secho(f"❌ Error loading dataset file: {e}", fg='red', err=True)
        sys.exit(1)

    # 2. Initialize Data Processors & Transformers
    processor = DataProcessor()
    transformer = EnergyTransformer()
    
    # 10 fixed random seeds to guarantee scientific replication & reproducibility
    seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    accuracy_list = []

    click.echo(f"-> Configuration: K = {cluster} clusters | C = {c_param} | Test Size = {test_size}")
    click.echo("----------------------------------------------------------------------")

    # 3. Execute 10-Run Experimental Evaluation Loop
    for run, seed in enumerate(seeds):
        # Adaptive data processing & independent clustering per seed
        X_train_scaled, X_test_scaled, y_train, y_test, classes, clusters_population = processor.process(
            df, test_size=test_size, n_energy_clusters=cluster, random_state=seed
        )
        
        # Energy-based space mapping/transformation
        X_train_energy = transformer.transform(X_train_scaled, clusters_population)
        X_test_energy = transformer.transform(X_test_scaled, clusters_population)
        
        # Standardize features in the computed energy space
        e_scaler = StandardScaler()
        X_train_energy = e_scaler.fit_transform(X_train_energy)
        X_test_energy = e_scaler.transform(X_test_energy)
        
        # Train Support Vector Classifier (SVC)
        trainer = ModelTrainer(C=c_param, kernel='rbf', gamma='scale', random_state=seed)
        trainer.train(X_train_energy, y_train)
        
        # Predict & Evaluate performance metrics
        y_pred = trainer.predict(X_test_energy)
        acc, cm = Evaluator.get_results(y_test, y_pred)
        accuracy_list.append(acc)
        
        # Output progress log for individual runs
        click.echo(f" Run {run + 1:02d}: Accuracy = {acc * 100:.2f} %")

    # 4. Compute & Display Final Aggregated Statistical Summary
    mean_accuracy = np.mean(accuracy_list)
    std_accuracy = np.std(accuracy_list)
    
    click.secho("\n======================================================================", fg='cyan')
    click.secho("📈 FINAL AGGREGATED STATISTICAL RESULTS (AFTER 10 RUNS)", fg='yellow', bold=True)
    click.secho(f" MEAN ACCURACY: {mean_accuracy * 100:.2f} %", fg='green', bold=True)
    click.secho(f" STANDARD DEVIATION (Std): {std_accuracy * 100:.2f} %", fg='magenta')
    click.secho("======================================================================\n", fg='cyan')

if __name__ == '__main__':
    main()