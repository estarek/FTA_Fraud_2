import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Set page configuration
st.set_page_config(
    page_title="E-Invoice Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define paths
MODEL_DIR = "model_artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_detection_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
LABEL_ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")
FEATURE_METADATA_PATH = os.path.join(MODEL_DIR, "feature_metadata.json")
EVALUATION_METRICS_PATH = os.path.join(MODEL_DIR, "evaluation_metrics.json")
RISK_SCORED_INVOICES_PATH = os.path.join(MODEL_DIR, "risk_scored_invoices.csv")
TOP_RISK_INVOICES_PATH = os.path.join(MODEL_DIR, "top_risk_invoices.csv")
BOTTOM_RISK_INVOICES_PATH = os.path.join(MODEL_DIR, "bottom_risk_invoices.csv")
ANOMALY_TYPE_DISTRIBUTION_PATH = os.path.join(MODEL_DIR, "anomaly_type_distribution.csv")
EMIRATE_DISTRIBUTION_PATH = os.path.join(MODEL_DIR, "emirate_distribution.csv")

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1E3A8A;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 1rem;
        color: #4B5563;
    }
    .high-risk {
        color: #DC2626;
        font-weight: bold;
    }
    .medium-risk {
        color: #F59E0B;
        font-weight: bold;
    }
    .low-risk {
        color: #10B981;
        font-weight: bold;
    }
    .info-box {
        background-color: #EFF6FF;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #3B82F6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A;
        color: white;
    }
    .error-box {
        background-color: #FEE2E2;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #DC2626;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def load_data():
    """Load all necessary data and model artifacts with robust error handling"""
    data = {}
    missing_files = []
    
    # Define all files to load
    files_to_load = {
        "model": MODEL_PATH,
        "scaler": SCALER_PATH,
        "label_encoders": LABEL_ENCODERS_PATH,
        "feature_metadata": FEATURE_METADATA_PATH,
        "evaluation_metrics": EVALUATION_METRICS_PATH,
        "risk_scored_invoices": RISK_SCORED_INVOICES_PATH,
        "top_risk_invoices": TOP_RISK_INVOICES_PATH,
        "bottom_risk_invoices": BOTTOM_RISK_INVOICES_PATH,
        "anomaly_type_distribution": ANOMALY_TYPE_DISTRIBUTION_PATH,
        "emirate_distribution": EMIRATE_DISTRIBUTION_PATH
    }
    
    # Check if model_artifacts directory exists
    if not os.path.exists(MODEL_DIR):
        st.error(f"Model artifacts directory '{MODEL_DIR}' not found. Please ensure it exists in the same directory as this script.")
        return None
    
    # Check which files exist
    for key, file_path in files_to_load.items():
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        st.error(f"The following required files are missing: {', '.join(missing_files)}")
        return None
    
    # Load files with appropriate error handling
    try:
        # Load model
        try:
            data["model"] = joblib.load(MODEL_PATH)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            data["model"] = None
        
        # Load scaler
        try:
            data["scaler"] = joblib.load(SCALER_PATH)
        except Exception as e:
            st.error(f"Error loading scaler: {e}")
            data["scaler"] = None
        
        # Load label encoders
        try:
            with open(LABEL_ENCODERS_PATH, 'rb') as f:
                data["label_encoders"] = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading label encoders: {e}")
            data["label_encoders"] = None
        
        # Load feature metadata
        try:
            with open(FEATURE_METADATA_PATH, 'r') as f:
                data["feature_metadata"] = json.load(f)
        except Exception as e:
            st.error(f"Error loading feature metadata: {e}")
            data["feature_metadata"] = None
        
        # Load evaluation metrics
        try:
            with open(EVALUATION_METRICS_PATH, 'r') as f:
                data["evaluation_metrics"] = json.load(f)
        except Exception as e:
            st.error(f"Error loading evaluation metrics: {e}")
            data["evaluation_metrics"] = None
        
        # Load CSV files with error handling
        csv_files = {
            "risk_scored_invoices": RISK_SCORED_INVOICES_PATH,
            "top_risk_invoices": TOP_RISK_INVOICES_PATH,
            "bottom_risk_invoices": BOTTOM_RISK_INVOICES_PATH,
            "anomaly_type_distribution": ANOMALY_TYPE_DISTRIBUTION_PATH,
            "emirate_distribution": EMIRATE_DISTRIBUTION_PATH
        }
        
        for key, file_path in csv_files.items():
            try:
                data[key] = pd.read_csv(file_path)
            except Exception as e:
                st.error(f"Error loading {key}: {e}")
                # Create empty DataFrame with expected columns as fallback
                if key == "risk_scored_invoices" or key == "top_risk_invoices" or key == "bottom_risk_invoices":
                    data[key] = pd.DataFrame(columns=["invoice_number", "true_anomaly", "anomaly_risk_score", 
                                                     "predicted_anomaly", "original_anomaly_type", "original_explanation"])
                elif key == "anomaly_type_distribution":
                    data[key] = pd.DataFrame(columns=["anomaly_type", "count"])
                elif key == "emirate_distribution":
                    data[key] = pd.DataFrame(columns=["emirate", "count"])
        
        return data
    
    except Exception as e:
        st.error(f"Unexpected error loading data: {e}")
        return None

@st.cache_data
def plot_risk_score_distribution(risk_scored_invoices):
    """Plot the distribution of risk scores with error handling"""
    try:
        if risk_scored_invoices is None or len(risk_scored_invoices) == 0:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for risk score distribution",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Distribution of Risk Scores (No Data)",
                height=400,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig
        
        # Check if required columns exist
        required_cols = ["anomaly_risk_score", "true_anomaly"]
        if not all(col in risk_scored_invoices.columns for col in required_cols):
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="Missing required columns for risk score distribution",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Distribution of Risk Scores (Missing Data)",
                height=400,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig
        
        # Ensure true_anomaly is numeric
        risk_scored_invoices = risk_scored_invoices.copy()
        risk_scored_invoices["true_anomaly"] = pd.to_numeric(risk_scored_invoices["true_anomaly"], errors="coerce").fillna(0).astype(int)
        
        fig = px.histogram(
            risk_scored_invoices, 
            x="anomaly_risk_score",
            color="true_anomaly",
            nbins=50,
            labels={"anomaly_risk_score": "Risk Score", "true_anomaly": "Is Anomaly"},
            color_discrete_map={0: "#10B981", 1: "#DC2626"},
            title="Distribution of Risk Scores"
        )
        
        fig.update_layout(
            xaxis_title="Risk Score",
            yaxis_title="Count",
            legend_title="Is Anomaly",
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        
        return fig
    except Exception as e:
        # Return error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error plotting risk score distribution: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(
            title="Distribution of Risk Scores (Error)",
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig

@st.cache_data
def plot_confusion_matrix(evaluation_metrics):
    """Plot the confusion matrix with robust error handling"""
    try:
        if evaluation_metrics is None or "confusion_matrix" not in evaluation_metrics:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No confusion matrix data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Confusion Matrix (No Data)",
                height=400,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig
        
        # Get confusion matrix and convert to numpy array if it's a list
        cm = evaluation_metrics["confusion_matrix"]
        
        # Convert to numpy array if it's a list
        if isinstance(cm, list):
            cm = np.array(cm)
        
        # Calculate percentages - handle both list and numpy array formats
        total = float(np.sum(cm) if isinstance(cm, np.ndarray) else sum(sum(row) for row in cm))
        
        if isinstance(cm, np.ndarray):
            cm_percent = [[val / total * 100 for val in row] for row in cm]
        else:
            cm_percent = [[val / total * 100 for val in row] for row in cm]
        
        # Create annotation text
        annotations = []
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                annotations.append(
                    dict(
                        x=j,
                        y=i,
                        text=f"{cm[i][j]}<br>({cm_percent[i][j]:.1f}%)",
                        font=dict(color="white" if (i == j) else "black"),
                        showarrow=False,
                    )
                )
        
        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=["Predicted Normal", "Predicted Anomaly"],
            y=["Actual Normal", "Actual Anomaly"],
            colorscale=[[0, "#10B981"], [1, "#DC2626"]],
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            annotations=annotations,
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        
        return fig
    except Exception as e:
        # Return error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error plotting confusion matrix: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(
            title="Confusion Matrix (Error)",
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig

@st.cache_data
def plot_feature_importance(model, feature_metadata):
    """Plot feature importance with error handling"""
    try:
        if model is None or feature_metadata is None or "valid_features" not in feature_metadata:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No feature importance data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Feature Importance (No Data)",
                height=500,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig
        
        # Get feature importance
        try:
            feature_importance = model.feature_importances_
        except AttributeError:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="Model does not have feature_importances_ attribute",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Feature Importance (Unsupported Model)",
                height=500,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig
        
        # Check if lengths match
        if len(feature_importance) != len(feature_metadata["valid_features"]):
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Feature importance length ({len(feature_importance)}) doesn't match feature names length ({len(feature_metadata['valid_features'])})",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Feature Importance (Dimension Mismatch)",
                height=500,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig
        
        # Create a dataframe for plotting
        feature_importance_df = pd.DataFrame({
            "Feature": feature_metadata["valid_features"],
            "Importance": feature_importance
        })
        
        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values("Importance", ascending=False).head(15)
        
        # Create the plot
        fig = px.bar(
            feature_importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 15 Feature Importance",
            color="Importance",
            color_continuous_scale=px.colors.sequential.Blues,
        )
        
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            height=500,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        
        return fig
    except Exception as e:
        # Return error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error plotting feature importance: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(
            title="Feature Importance (Error)",
            height=500,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig

@st.cache_data
def plot_anomaly_type_distribution(anomaly_type_distribution):
    """Plot the distribution of anomaly types with error handling"""
    try:
        if anomaly_type_distribution is None or len(anomaly_type_distribution) == 0:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No anomaly type distribution data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Anomaly Type Distribution (No Data)",
                height=500,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig
        
        # Check if required columns exist
        required_cols = ["anomaly_type", "count"]
        if not all(col in anomaly_type_distribution.columns for col in required_cols):
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="Missing required columns for anomaly type distribution",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Anomaly Type Distribution (Missing Data)",
                height=500,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig
        
        # Sort by count
        anomaly_type_distribution = anomaly_type_distribution.sort_values("count", ascending=False)
        
        # Create the plot
        fig = px.bar(
            anomaly_type_distribution,
            x="count",
            y="anomaly_type",
            orientation="h",
            title="Anomaly Type Distribution",
            color="count",
            color_continuous_scale=px.colors.sequential.Reds,
        )
        
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            height=500,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        
        return fig
    except Exception as e:
        # Return error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error plotting anomaly type distribution: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(
            title="Anomaly Type Distribution (Error)",
            height=500,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig

@st.cache_data
def plot_emirate_distribution(emirate_distribution):
    """Plot the distribution of emirates with error handling"""
    try:
        if emirate_distribution is None or len(emirate_distribution) == 0:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No emirate distribution data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Invoice Distribution by Emirate (No Data)",
                height=400,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig
        
        # Check if required columns exist
        required_cols = ["emirate", "count"]
        if not all(col in emirate_distribution.columns for col in required_cols):
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="Missing required columns for emirate distribution",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Invoice Distribution by Emirate (Missing Data)",
                height=400,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig
        
        # Create the plot
        fig = px.pie(
            emirate_distribution,
            values="count",
            names="emirate",
            title="Invoice Distribution by Emirate",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        
        return fig
    except Exception as e:
        # Return error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error plotting emirate distribution: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(
            title="Invoice Distribution by Emirate (Error)",
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig

@st.cache_data
def plot_risk_by_emirate(risk_scored_invoices, emirate_distribution):
    """Plot the average risk score by emirate with error handling"""
    try:
        if risk_scored_invoices is None or emirate_distribution is None or len(risk_scored_invoices) == 0 or len(emirate_distribution) == 0:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for risk by emirate",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Average Risk Score by Emirate (No Data)",
                height=400,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig
        
        # Check if required columns exist
        if "buyer_emirate" not in risk_scored_invoices.columns or "anomaly_risk_score" not in risk_scored_invoices.columns:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="Missing required columns in risk_scored_invoices",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Average Risk Score by Emirate (Missing Data)",
                height=400,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig
        
        if "emirate" not in emirate_distribution.columns or "count" not in emirate_distribution.columns:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="Missing required columns in emirate_distribution",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Average Risk Score by Emirate (Missing Data)",
                height=400,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig
        
        # Calculate average risk score by emirate
        risk_by_emirate = risk_scored_invoices.groupby("buyer_emirate")["anomaly_risk_score"].mean().reset_index()
        risk_by_emirate.columns = ["emirate", "avg_risk_score"]
        
        # Merge with emirate distribution to get counts
        risk_by_emirate = pd.merge(risk_by_emirate, emirate_distribution, on="emirate", how="inner")
        
        # Check if we have data after merging
        if len(risk_by_emirate) == 0:
            # Return empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No matching data between risk scores and emirate distribution",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title="Average Risk Score by Emirate (No Matching Data)",
                height=400,
                margin=dict(l=40, r=40, t=40, b=40),
            )
            return fig
        
        # Create the plot
        fig = px.scatter(
            risk_by_emirate,
            x="avg_risk_score",
            y="count",
            size="count",
            color="avg_risk_score",
            hover_name="emirate",
            color_continuous_scale=px.colors.sequential.Reds,
            title="Average Risk Score by Emirate",
            labels={"avg_risk_score": "Average Risk Score", "count": "Number of Invoices"},
        )
        
        fig.update_layout(
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        
        return fig
    except Exception as e:
        # Return error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error plotting risk by emirate: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        fig.update_layout(
            title="Average Risk Score by Emirate (Error)",
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
        )
        return fig

def format_risk_score(score):
    """Format risk score with color coding and error handling"""
    try:
        score_float = float(score)
        if score_float >= 0.7:
            return f'<span class="high-risk">{score_float:.2f}</span>'
        elif score_float >= 0.3:
            return f'<span class="medium-risk">{score_float:.2f}</span>'
        else:
            return f'<span class="low-risk">{score_float:.2f}</span>'
    except (ValueError, TypeError):
        return f'<span class="medium-risk">N/A</span>'

def main():
    # Load data with error handling
    data = load_data()
    
    if data is None:
        st.error("Failed to load data. Please make sure the model artifacts are available.")
        
        st.markdown("""
        <div class="error-box">
            <h3>Troubleshooting Steps:</h3>
            <ol>
                <li>Ensure the 'model_artifacts' directory exists in the same directory as this script.</li>
                <li>Check that all required files are present in the model_artifacts directory.</li>
                <li>Verify that you have the necessary permissions to read the files.</li>
                <li>If running in Streamlit Cloud, make sure all files were properly uploaded to the repository.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        return
    
    # Extract data with error handling
    model = data.get("model")
    evaluation_metrics = data.get("evaluation_metrics", {})
    risk_scored_invoices = data.get("risk_scored_invoices", pd.DataFrame())
    top_risk_invoices = data.get("top_risk_invoices", pd.DataFrame())
    bottom_risk_invoices = data.get("bottom_risk_invoices", pd.DataFrame())
    anomaly_type_distribution = data.get("anomaly_type_distribution", pd.DataFrame())
    emirate_distribution = data.get("emirate_distribution", pd.DataFrame())
    feature_metadata = data.get("feature_metadata", {})
    
    # Header
    st.markdown('<h1 class="main-header">E-Invoice Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        This dashboard demonstrates AI capabilities for fraud detection and risk scoring in e-invoices.
        The model analyzes patterns in invoice data to identify potential anomalies and assigns risk scores.
        Explore the dashboard to see model performance, risk distributions, and detailed invoice analysis.
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Risk Analysis", "🌍 Geographic Insights", "📝 Invoice Explorer"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Model Performance</h2>', unsafe_allow_html=True)
        
        # Key metrics with error handling
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            accuracy = evaluation_metrics.get("accuracy", 0)
            if not isinstance(accuracy, (int, float)):
                accuracy = 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{accuracy * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            try:
                precision = evaluation_metrics.get("classification_report", {}).get("Anomaly", {}).get("precision", 0)
                if not isinstance(precision, (int, float)):
                    precision = 0
            except (KeyError, AttributeError, TypeError):
                precision = 0
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Precision (Anomaly)</div>
                <div class="metric-value">{precision * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            try:
                recall = evaluation_metrics.get("classification_report", {}).get("Anomaly", {}).get("recall", 0)
                if not isinstance(recall, (int, float)):
                    recall = 0
            except (KeyError, AttributeError, TypeError):
                recall = 0
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Recall (Anomaly)</div>
                <div class="metric-value">{recall * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            try:
                f1 = evaluation_metrics.get("classification_report", {}).get("Anomaly", {}).get("f1-score", 0)
                if not isinstance(f1, (int, float)):
                    f1 = 0
            except (KeyError, AttributeError, TypeError):
                f1 = 0
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">F1 Score (Anomaly)</div>
                <div class="metric-value">{f1 * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Confusion matrix and risk score distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plot_confusion_matrix(evaluation_metrics), use_container_width=True)
        
        with col2:
            st.plotly_chart(plot_risk_score_distribution(risk_scored_invoices), use_container_width=True)
        
        # Feature importance
        st.plotly_chart(plot_feature_importance(model, feature_metadata), use_container_width=True)
        
        # Anomaly type distribution
        st.plotly_chart(plot_anomaly_type_distribution(anomaly_type_distribution), use_container_width=True)
    
    with tab2:
        st.markdown('<h2 class="sub-header">Risk Score Analysis</h2>', unsafe_allow_html=True)
        
        # Risk score threshold slider
        risk_threshold = st.slider(
            "Risk Score Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Adjust the threshold to see how it affects the classification of invoices."
        )
        
        # Calculate metrics based on threshold with error handling
        try:
            if "anomaly_risk_score" in risk_scored_invoices.columns and "true_anomaly" in risk_scored_invoices.columns:
                # Ensure numeric types
                risk_scored_invoices_clean = risk_scored_invoices.copy()
                risk_scored_invoices_clean["anomaly_risk_score"] = pd.to_numeric(risk_scored_invoices_clean["anomaly_risk_score"], errors="coerce").fillna(0)
                risk_scored_invoices_clean["true_anomaly"] = pd.to_numeric(risk_scored_invoices_clean["true_anomaly"], errors="coerce").fillna(0).astype(int)
                
                predicted_anomaly = (risk_scored_invoices_clean["anomaly_risk_score"] >= risk_threshold).astype(int)
                true_anomaly = risk_scored_invoices_clean["true_anomaly"]
                
                # Calculate confusion matrix
                cm = confusion_matrix(true_anomaly, predicted_anomaly)
                tn, fp, fn, tp = cm.ravel()
                
                # Calculate metrics
                accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            else:
                # Default values if columns are missing
                accuracy, precision, recall, f1 = 0, 0, 0, 0
                cm = np.array([[0, 0], [0, 0]])
                tn, fp, fn, tp = 0, 0, 0, 0
        except Exception as e:
            st.error(f"Error calculating metrics: {e}")
            accuracy, precision, recall, f1 = 0, 0, 0, 0
            cm = np.array([[0, 0], [0, 0]])
            tn, fp, fn, tp = 0, 0, 0, 0
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{accuracy * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Precision</div>
                <div class="metric-value">{precision * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Recall</div>
                <div class="metric-value">{recall * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">F1 Score</div>
                <div class="metric-value">{f1 * 100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display confusion matrix
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                # Create annotation text
                total = float(np.sum(cm))
                if total > 0:
                    cm_percent = [[val / total * 100 for val in row] for row in cm]
                else:
                    cm_percent = [[0.0 for _ in row] for row in cm]
                
                annotations = []
                for i in range(len(cm)):
                    for j in range(len(cm[i])):
                        annotations.append(
                            dict(
                                x=j,
                                y=i,
                                text=f"{cm[i][j]}<br>({cm_percent[i][j]:.1f}%)",
                                font=dict(color="white" if (i == j) else "black"),
                                showarrow=False,
                            )
                        )
                
                # Create the heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=["Predicted Normal", "Predicted Anomaly"],
                    y=["Actual Normal", "Actual Anomaly"],
                    colorscale=[[0, "#10B981"], [1, "#DC2626"]],
                ))
                
                fig.update_layout(
                    title=f"Confusion Matrix (Threshold: {risk_threshold:.2f})",
                    annotations=annotations,
                    height=400,
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying confusion matrix: {e}")
                # Fallback empty figure
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Error displaying confusion matrix: {str(e)}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                fig.update_layout(
                    title=f"Confusion Matrix (Error)",
                    height=400,
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            try:
                # Create ROC curve if we have the necessary data
                if "true_anomaly" in risk_scored_invoices.columns and "anomaly_risk_score" in risk_scored_invoices.columns:
                    from sklearn.metrics import roc_curve, auc
                    
                    # Ensure numeric types
                    risk_scored_invoices_clean = risk_scored_invoices.copy()
                    risk_scored_invoices_clean["anomaly_risk_score"] = pd.to_numeric(risk_scored_invoices_clean["anomaly_risk_score"], errors="coerce").fillna(0)
                    risk_scored_invoices_clean["true_anomaly"] = pd.to_numeric(risk_scored_invoices_clean["true_anomaly"], errors="coerce").fillna(0).astype(int)
                    
                    fpr, tpr, thresholds = roc_curve(risk_scored_invoices_clean["true_anomaly"], risk_scored_invoices_clean["anomaly_risk_score"])
                    roc_auc = auc(fpr, tpr)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'ROC Curve (AUC = {roc_auc:.3f})',
                        line=dict(color='#1E3A8A', width=2)
                    ))
                    
                    # Add current threshold point
                    if (fp + tn) > 0 and (tp + fn) > 0:
                        current_fpr = fp / (fp + tn)
                        current_tpr = tp / (tp + fn)
                        
                        fig.add_trace(go.Scatter(
                            x=[current_fpr], y=[current_tpr],
                            mode='markers',
                            name=f'Current Threshold ({risk_threshold:.2f})',
                            marker=dict(color='red', size=10)
                        ))
                    
                    # Add diagonal line
                    fig.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='Random Classifier',
                        line=dict(color='gray', width=2, dash='dash')
                    ))
                    
                    fig.update_layout(
                        title='ROC Curve',
                        xaxis=dict(title='False Positive Rate'),
                        yaxis=dict(title='True Positive Rate'),
                        height=400,
                        margin=dict(l=40, r=40, t=40, b=40),
                        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0.5)'),
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Fallback empty figure
                    fig = go.Figure()
                    fig.add_annotation(
                        text="Missing data for ROC curve",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )
                    fig.update_layout(
                        title='ROC Curve (Missing Data)',
                        height=400,
                        margin=dict(l=40, r=40, t=40, b=40),
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying ROC curve: {e}")
                # Fallback empty figure
                fig = go.Figure()
                fig.add_annotation(
                    text=f"Error displaying ROC curve: {str(e)}",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                fig.update_layout(
                    title='ROC Curve (Error)',
                    height=400,
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Display top high-risk invoices
        st.markdown('<h3 class="sub-header">Top High-Risk Invoices</h3>', unsafe_allow_html=True)
        
        try:
            # Filter invoices based on threshold
            if "anomaly_risk_score" in risk_scored_invoices.columns:
                risk_scored_invoices_clean = risk_scored_invoices.copy()
                risk_scored_invoices_clean["anomaly_risk_score"] = pd.to_numeric(risk_scored_invoices_clean["anomaly_risk_score"], errors="coerce").fillna(0)
                
                high_risk_invoices = risk_scored_invoices_clean[risk_scored_invoices_clean["anomaly_risk_score"] >= risk_threshold].sort_values("anomaly_risk_score", ascending=False).head(10)
                
                if len(high_risk_invoices) > 0:
                    # Format the dataframe for display
                    display_cols = ["invoice_number", "anomaly_risk_score", "true_anomaly", "original_anomaly_type", "original_explanation"]
                    display_cols = [col for col in display_cols if col in high_risk_invoices.columns]
                    
                    if len(display_cols) > 0:
                        display_df = high_risk_invoices[display_cols].copy()
                        
                        # Format columns if they exist
                        if "anomaly_risk_score" in display_df.columns:
                            display_df["anomaly_risk_score"] = display_df["anomaly_risk_score"].apply(lambda x: f"{float(x):.2f}" if pd.notnull(x) else "N/A")
                        
                        if "true_anomaly" in display_df.columns:
                            display_df["true_anomaly"] = display_df["true_anomaly"].apply(lambda x: "Yes" if x == 1 or x == "1" or x == "True" or x is True else "No")
                        
                        # Rename columns
                        column_mapping = {
                            "invoice_number": "Invoice Number",
                            "anomaly_risk_score": "Risk Score",
                            "true_anomaly": "Is Anomaly",
                            "original_anomaly_type": "Anomaly Type",
                            "original_explanation": "Explanation"
                        }
                        
                        display_df.columns = [column_mapping.get(col, col) for col in display_df.columns]
                        
                        st.dataframe(display_df, use_container_width=True)
                    else:
                        st.info("No display columns available in the data.")
                else:
                    st.info("No invoices found above the current risk threshold.")
            else:
                st.info("Risk score data is not available.")
        except Exception as e:
            st.error(f"Error displaying high-risk invoices: {e}")
    
    with tab3:
        st.markdown('<h2 class="sub-header">Geographic Distribution</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(plot_emirate_distribution(emirate_distribution), use_container_width=True)
        
        with col2:
            st.plotly_chart(plot_risk_by_emirate(risk_scored_invoices, emirate_distribution), use_container_width=True)
        
        # Create a choropleth map of UAE emirates
        st.markdown('<h3 class="sub-header">Risk Score Map by Emirate</h3>', unsafe_allow_html=True)
        
        try:
            # Calculate average risk score by emirate
            if "buyer_emirate" in risk_scored_invoices.columns and "anomaly_risk_score" in risk_scored_invoices.columns:
                risk_scored_invoices_clean = risk_scored_invoices.copy()
                risk_scored_invoices_clean["anomaly_risk_score"] = pd.to_numeric(risk_scored_invoices_clean["anomaly_risk_score"], errors="coerce").fillna(0)
                
                risk_by_emirate = risk_scored_invoices_clean.groupby("buyer_emirate")["anomaly_risk_score"].mean().reset_index()
                risk_by_emirate.columns = ["emirate", "avg_risk_score"]
                
                # UAE emirates with approximate coordinates
                emirates_coords = {
                    "Abu Dhabi": {"lat": 24.4539, "lon": 54.3773},
                    "Dubai": {"lat": 25.2048, "lon": 55.2708},
                    "Sharjah": {"lat": 25.3463, "lon": 55.4209},
                    "Ajman": {"lat": 25.4111, "lon": 55.4354},
                    "Umm Al Quwain": {"lat": 25.5647, "lon": 55.5534},
                    "Ras Al Khaimah": {"lat": 25.7895, "lon": 55.9432},
                    "Fujairah": {"lat": 25.1288, "lon": 56.3265}
                }
                
                # Create a dataframe for the map
                map_data = []
                for emirate, coords in emirates_coords.items():
                    risk_score = risk_by_emirate[risk_by_emirate["emirate"] == emirate]["avg_risk_score"].values
                    if len(risk_score) > 0:
                        map_data.append({
                            "emirate": emirate,
                            "lat": coords["lat"],
                            "lon": coords["lon"],
                            "avg_risk_score": risk_score[0]
                        })
                    else:
                        map_data.append({
                            "emirate": emirate,
                            "lat": coords["lat"],
                            "lon": coords["lon"],
                            "avg_risk_score": 0
                        })
                
                map_df = pd.DataFrame(map_data)
                
                # Create the map
                fig = px.scatter_mapbox(
                    map_df,
                    lat="lat",
                    lon="lon",
                    color="avg_risk_score",
                    size="avg_risk_score",
                    hover_name="emirate",
                    color_continuous_scale=px.colors.sequential.Reds,
                    size_max=15,
                    zoom=6,
                    mapbox_style="carto-positron",
                    title="Average Risk Score by Emirate",
                    labels={"avg_risk_score": "Average Risk Score"}
                )
                
                fig.update_layout(
                    height=600,
                    margin=dict(l=0, r=0, t=40, b=0),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Geographic data is not available for mapping.")
        except Exception as e:
            st.error(f"Error creating geographic map: {e}")
            # Fallback empty figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating geographic map: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(
                title='Geographic Map (Error)',
                height=600,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Add a note about the map
        st.markdown("""
        <div class="info-box">
            <strong>Note:</strong> The map shows the average risk score for each emirate. 
            Larger and darker circles indicate higher average risk scores.
            This visualization helps identify geographic patterns in anomalous invoices.
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<h2 class="sub-header">Invoice Explorer</h2>', unsafe_allow_html=True)
        
        # Search by invoice number
        st.markdown("### Search by Invoice Number")
        invoice_number = st.text_input("Enter Invoice Number")
        
        if invoice_number:
            try:
                # Search for the invoice
                if "invoice_number" in risk_scored_invoices.columns:
                    invoice = risk_scored_invoices[risk_scored_invoices["invoice_number"] == invoice_number]
                    
                    if len(invoice) > 0:
                        # Display invoice details
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Invoice Number</div>
                                <div class="metric-value">{invoice["invoice_number"].values[0]}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            if "anomaly_risk_score" in invoice.columns:
                                risk_score = invoice["anomaly_risk_score"].values[0]
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">Risk Score</div>
                                    <div class="metric-value">{format_risk_score(risk_score)}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div class="metric-card">
                                    <div class="metric-label">Risk Score</div>
                                    <div class="metric-value">N/A</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col3:
                            if "true_anomaly" in invoice.columns:
                                is_anomaly = "Yes" if invoice["true_anomaly"].values[0] == 1 or invoice["true_anomaly"].values[0] == "1" or invoice["true_anomaly"].values[0] == "True" or invoice["true_anomaly"].values[0] is True else "No"
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-label">Is Anomaly</div>
                                    <div class="metric-value">{is_anomaly}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div class="metric-card">
                                    <div class="metric-label">Is Anomaly</div>
                                    <div class="metric-value">N/A</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Display anomaly details if it's an anomaly
                        if "true_anomaly" in invoice.columns and (invoice["true_anomaly"].values[0] == 1 or invoice["true_anomaly"].values[0] == "1" or invoice["true_anomaly"].values[0] == "True" or invoice["true_anomaly"].values[0] is True):
                            st.markdown("### Anomaly Details")
                            
                            anomaly_type = invoice["original_anomaly_type"].values[0] if "original_anomaly_type" in invoice.columns else "Unknown"
                            explanation = invoice["original_explanation"].values[0] if "original_explanation" in invoice.columns else "No explanation available"
                            
                            st.markdown(f"""
                            <div class="info-box">
                                <strong>Anomaly Type:</strong> {anomaly_type}<br>
                                <strong>Explanation:</strong> {explanation}
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning(f"Invoice {invoice_number} not found.")
                else:
                    st.warning("Invoice number column not found in the data.")
            except Exception as e:
                st.error(f"Error searching for invoice: {e}")
        
        # Browse invoices
        st.markdown("### Browse Invoices")
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            risk_level = st.selectbox(
                "Risk Level",
                ["All", "High Risk (>= 0.7)", "Medium Risk (0.3 - 0.7)", "Low Risk (< 0.3)"]
            )
        
        with col2:
            is_anomaly = st.selectbox(
                "Is Anomaly",
                ["All", "Yes", "No"]
            )
        
        try:
            # Apply filters
            filtered_invoices = risk_scored_invoices.copy()
            
            # Ensure numeric risk score
            if "anomaly_risk_score" in filtered_invoices.columns:
                filtered_invoices["anomaly_risk_score"] = pd.to_numeric(filtered_invoices["anomaly_risk_score"], errors="coerce").fillna(0)
                
                if risk_level == "High Risk (>= 0.7)":
                    filtered_invoices = filtered_invoices[filtered_invoices["anomaly_risk_score"] >= 0.7]
                elif risk_level == "Medium Risk (0.3 - 0.7)":
                    filtered_invoices = filtered_invoices[(filtered_invoices["anomaly_risk_score"] >= 0.3) & (filtered_invoices["anomaly_risk_score"] < 0.7)]
                elif risk_level == "Low Risk (< 0.3)":
                    filtered_invoices = filtered_invoices[filtered_invoices["anomaly_risk_score"] < 0.3]
            
            # Filter by anomaly status
            if "true_anomaly" in filtered_invoices.columns:
                if is_anomaly == "Yes":
                    filtered_invoices = filtered_invoices[filtered_invoices["true_anomaly"].astype(str).isin(["1", "True", "true"])]
                elif is_anomaly == "No":
                    filtered_invoices = filtered_invoices[~filtered_invoices["true_anomaly"].astype(str).isin(["1", "True", "true"])]
            
            # Sort by risk score
            if "anomaly_risk_score" in filtered_invoices.columns:
                filtered_invoices = filtered_invoices.sort_values("anomaly_risk_score", ascending=False)
            
            # Display invoices
            if len(filtered_invoices) > 0:
                # Format the dataframe for display
                display_cols = ["invoice_number", "anomaly_risk_score", "true_anomaly", "original_anomaly_type", "original_explanation"]
                display_cols = [col for col in display_cols if col in filtered_invoices.columns]
                
                if len(display_cols) > 0:
                    display_df = filtered_invoices[display_cols].head(100).copy()
                    
                    # Format columns
                    if "anomaly_risk_score" in display_df.columns:
                        display_df["risk_score_formatted"] = display_df["anomaly_risk_score"].apply(lambda x: format_risk_score(x))
                        display_df = display_df.drop(columns=["anomaly_risk_score"])
                    
                    if "true_anomaly" in display_df.columns:
                        display_df["true_anomaly"] = display_df["true_anomaly"].apply(lambda x: "Yes" if x == 1 or x == "1" or x == "True" or x is True else "No")
                    
                    # Rename columns
                    column_mapping = {
                        "invoice_number": "Invoice Number",
                        "risk_score_formatted": "Risk Score",
                        "true_anomaly": "Is Anomaly",
                        "original_anomaly_type": "Anomaly Type",
                        "original_explanation": "Explanation"
                    }
                    
                    display_df.columns = [column_mapping.get(col, col) for col in display_df.columns]
                    
                    st.markdown(f"Showing top 100 of {len(filtered_invoices)} invoices")
                    st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                else:
                    st.info("No display columns available in the data.")
            else:
                st.info("No invoices found matching the selected filters.")
        except Exception as e:
            st.error(f"Error filtering invoices: {e}")

if __name__ == "__main__":
    main()
