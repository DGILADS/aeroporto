# Airport Operations Statistical Modeling

## 🛫 Overview

This project is a professional engineering and data analysis solution for managing airport operations. It leverages various statistical distributions to model real-world scenarios such as check-in times, emergency landings, and security procedures.

## 🎯 Objectives

- **Scalable Architecture**: Clean Architecture and SOLID principles.
- **Statistical Engineering**: Practical application of Bernoulli, Binomial, Normal, Poisson, Multinomial, and Geometric distributions.
- **Data Pipeline**: Automated ETL with cleaning, normalization, and analysis.
- **Interactive Dashboard**: Visual insights into operational efficiency.

## 🏗️ Architecture

The project follows a layered architecture:

- `core/`: Base abstractions and utilities.
- `domain/`: Statistical models and business logic.
- `services/`: Use cases and orchestration.
- `infrastructure/`: Data persistence and external integrations.
- `pipelines/`: Automated data processing.
- `dashboard/`: Interactive visualizations with Streamlit.

## 🛠️ Technologies

- **Python 3.10+**
- **Data Science**: Numpy, Pandas, Scipy, Scikit-learn, Statsmodels.
- **Visualization**: Matplotlib, Seaborn, Plotly, Streamlit.
- **Reliability**: Pytest, Typing.

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone git@github.com:DGILADS/aeroporto.git
   ```
2. Set up the virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   pip install -r requirements.txt
   ```
3. Run the dashboard:
   ```bash
   streamlit run src/dashboard/app.py
   ```

---

_Developed as a high-level portfolio project for Software Architecture and Data Engineering._
