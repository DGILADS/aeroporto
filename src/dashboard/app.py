import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.domain.normal import NormalDistribution
from src.domain.poisson import PoissonDistribution
from src.domain.discrete import BinomialDistribution
from src.services.analysis import CorrelationAnalyzer

st.set_page_config(page_title="Airport Operations Analytics", layout="wide")

st.title("🛫 Aeroporto Internacional: Insights Estatísticos")
st.markdown("""
Esta plataforma demonstra a aplicação de distribuições estatísticas na gestão de operações aeroportuárias complexas.
""")

# Sidebar settings
st.sidebar.header("Configurações de Simulação")
sample_size = st.sidebar.slider("Tamanho da Amostra", 100, 5000, 1000)

# Create layout
col1, col2 = st.columns(2)

with col1:
    st.header("⏳ Tempos de Check-in (Normal)")
    mean_checkin = st.slider("Média (min)", 5, 30, 15)
    std_checkin = st.slider("Desvio Padrão", 1, 10, 4)
    
    model_normal = NormalDistribution(mean=mean_checkin, std_dev=std_checkin)
    data_normal = model_normal.simulate(sample_size)
    
    st.pyplot(model_normal.plot(data_normal))

with col2:
    st.header("🚨 Emergências (Poisson)")
    lambda_val = st.slider("Lambda (λ - Eventos/Intervalo)", 0.1, 10.0, 2.5)
    
    model_poisson = PoissonDistribution(lam=lambda_val)
    data_poisson = model_poisson.simulate(sample_size)
    
    st.pyplot(model_poisson.plot(data_poisson))

st.divider()

col3, col4 = st.columns(2)

with col3:
    st.header("🛡️ Inspeções de Segurança (Binomial)")
    n_inspections = st.number_input("Número de Inspeções (n)", 1, 100, 20)
    p_success = st.slider("Probabilidade Sucesso (p)", 0.0, 1.0, 0.05)
    
    model_binom = BinomialDistribution(n=n_inspections, p=p_success)
    st.pyplot(model_binom.plot())

with col4:
    st.header("🔗 Correlação de Métricas")
    # Generate dummy related data for correlation demo
    df_corr = pd.DataFrame({
        'Checkin': data_normal,
        'Emergencies': data_poisson[:sample_size],
        'Security': model_binom.simulate(sample_size)
    })
    
    analyzer = CorrelationAnalyzer()
    corrs = analyzer.calculate_correlations(df_corr)
    st.pyplot(analyzer.plot_heatmap(corrs['pearson']))

st.sidebar.info(f"Dados processados: {sample_size} registros simulados.")
