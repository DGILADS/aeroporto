from src.domain.normal import NormalDistribution
from src.domain.poisson import PoissonDistribution
from src.domain.discrete import BinomialDistribution, BernoulliDistribution, GeometricDistribution, MultinomialDistribution
from src.pipelines.pipeline import DataPipeline
from src.services.analysis import CorrelationAnalyzer
import pandas as pd
import numpy as np

def run_simulation_flow():
    """Main orchestration of simulation, pipeline and analysis."""
    print("Starting Airport Operations Simulation...")
    
    # 1. Simulate data for each area
    normal_model = NormalDistribution(mean=15, std_dev=4) # 15 min check-in
    checkin_times = normal_model.simulate(1000)
    
    poisson_model = PoissonDistribution(lam=2.5) # 2.5 emergencies/month
    emergencies = poisson_model.simulate(1000)
    
    binom_model = BinomialDistribution(n=50, p=0.05) # 5% inspection success
    inspections = binom_model.simulate(1000)
    
    # Create a DataFrame
    df = pd.DataFrame({
        'checkin_time': checkin_times,
        'monthly_emergencies': emergencies,
        'security_detections': inspections
    })
    
    # 2. Pipeline processing
    pipeline = DataPipeline(
        raw_path="airport_operations/data/raw", 
        processed_path="airport_operations/data/processed"
    )
    
    pipeline.save_data(df, "airport_ops_raw.csv", is_raw=True)
    
    df_cleaned = pipeline.clean_data(df)
    df_normalized = pipeline.normalize(df_cleaned, method='zscore')
    
    pipeline.save_data(df_normalized, "airport_ops_processed.csv", is_raw=False)
    
    # 3. Analyze
    corrs = CorrelationAnalyzer.calculate_correlations(df_cleaned)
    print("\nPearson Correlation:")
    print(corrs['pearson'])
    
    print("\nSimulation and Pipeline Flow Completed Successfully!")

if __name__ == "__main__":
    run_simulation_flow()
