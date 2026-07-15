import pandas as pd
# pyrefly: ignore [missing-import]
import plotly.express as px
# pyrefly: ignore [missing-import]
import plotly.graph_objects as go
import json
from config import EMOTION_COLORS

def generate_emotion_distribution_chart(df):
    """Generates an interactive Pie Chart of overall emotion distributions."""
    if df.empty:
        return None
        
    counts = df["predicted_emotion"].value_counts().reset_index()
    counts.columns = ["Emotion", "Count"]
    
    fig = px.pie(
        counts, 
        values="Count", 
        names="Emotion",
        color="Emotion",
        color_discrete_map=EMOTION_COLORS,
        hole=0.4,
        title="<b>Overall Emotion Distribution</b>"
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0")
    )
    return fig

def generate_field_analysis_chart(df):
    """Generates a Grouped Bar Chart of emotion distributions across different academic fields."""
    if df.empty:
        return None
        
    # Group by field and emotion
    grouped = df.groupby(["field", "predicted_emotion"]).size().reset_index(name="Count")
    
    fig = px.bar(
        grouped,
        x="field",
        y="Count",
        color="predicted_emotion",
        color_discrete_map=EMOTION_COLORS,
        barmode="group",
        title="<b>Emotion Distribution by Academic Field</b>",
        labels={"field": "Academic Field", "Count": "Number of Records", "predicted_emotion": "Emotion"}
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0"),
        xaxis_tickangle=-45
    )
    return fig

def generate_confidence_trend_chart(df):
    """Generates a Line and Scatter chart plotting prediction confidence trends over time."""
    if df.empty:
        return None
        
    # Ensure timestamp is datetime
    df_copy = df.copy()
    df_copy["timestamp_dt"] = pd.to_datetime(df_copy["timestamp"])
    df_copy = df_copy.sort_values("timestamp_dt")
    
    fig = px.scatter(
        df_copy,
        x="timestamp_dt",
        y="confidence_score",
        color="predicted_emotion",
        color_discrete_map=EMOTION_COLORS,
        hover_data=["model_used", "field"],
        title="<b>Prediction Confidence Level Trends</b>",
        labels={"timestamp_dt": "Date & Time", "confidence_score": "Confidence Score", "predicted_emotion": "Emotion"}
    )
    
    # Add a trendline of rolling average confidence if there are enough points
    if len(df_copy) > 1:
        df_copy["rolling_avg"] = df_copy["confidence_score"].rolling(window=min(5, len(df_copy)), min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=df_copy["timestamp_dt"],
                y=df_copy["rolling_avg"],
                mode="lines",
                name="Rolling Average",
                line=dict(color="#FFD54F", width=2, dash="dash")
            )
        )
        
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0")
    )
    return fig

def generate_model_comparison_chart(df):
    """Generates a chart comparing the distribution of predictions by BERT vs BiLSTM."""
    if df.empty:
        return None
        
    grouped = df.groupby(["predicted_emotion", "model_used"]).size().reset_index(name="Count")
    
    fig = px.bar(
        grouped,
        x="predicted_emotion",
        y="Count",
        color="model_used",
        barmode="group",
        title="<b>Predictions Frequency Comparison: BiLSTM vs BERT</b>",
        labels={"predicted_emotion": "Predicted Emotion", "Count": "Prediction Count", "model_used": "Model"}
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0")
    )
    return fig
