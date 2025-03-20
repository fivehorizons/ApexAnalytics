import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
from utils import F1DataAPI, process_lap_times, calculate_advanced_statistics
import joblib
import os

def render_predictive_analytics(session_id):
    """Render the predictive analytics page with ML-powered insights."""
    st.header("Predictive Analytics")
    
    # Fetch lap times with error handling
    with st.spinner("Loading lap times..."):
        lap_times_df, error = F1DataAPI.get_lap_times(session_id)
        if error:
            st.error(f"Error loading lap times: {error}")
            return
            
        if lap_times_df is not None and not lap_times_df.empty:
            # Process lap times
            processed_laps = process_lap_times(lap_times_df)
            
            # Create tabs for different analyses
            tab1, tab2, tab3 = st.tabs([
                "🎯 Lap Time Prediction",
                "📈 Performance Trends",
                "🔄 Race Simulation"
            ])
            
            with tab1:
                render_lap_time_prediction(processed_laps)
            
            with tab2:
                render_performance_trends(processed_laps)
            
            with tab3:
                render_race_simulation(processed_laps)

def prepare_features(lap_data):
    """Prepare features for machine learning models."""
    # Create relevant features
    features = pd.DataFrame()
    
    # Basic features
    features['lap_number'] = lap_data['lap_number']
    features['driver_number'] = lap_data['driver_number']
    
    # Calculate rolling statistics
    for driver in lap_data['driver_number'].unique():
        driver_mask = lap_data['driver_number'] == driver
        features.loc[driver_mask, 'rolling_mean_3'] = (
            lap_data.loc[driver_mask, 'lap_time_seconds']
            .rolling(window=3, min_periods=1)
            .mean()
        )
        features.loc[driver_mask, 'rolling_std_3'] = (
            lap_data.loc[driver_mask, 'lap_time_seconds']
            .rolling(window=3, min_periods=1)
            .std()
        )
    
    # Fill NaN values with mean
    features = features.fillna(features.mean())
    
    return features

def train_prediction_model(features, target):
    """Train an optimized random forest model for lap time prediction."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, mse, r2, y_test, y_pred

def render_lap_time_prediction(lap_data):
    """Render lap time prediction analysis with ML model."""
    st.subheader("Lap Time Prediction")
    
    # Prepare data
    features = prepare_features(lap_data)
    target = lap_data['lap_time_seconds']
    
    # Train model and get predictions
    model, scaler, mse, r2, y_test, y_pred = train_prediction_model(features, target)
    
    # Display model performance
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Mean Squared Error",
            f"{mse:.4f}",
            "Lower is better"
        )
    
    with col2:
        st.metric(
            "R² Score",
            f"{r2:.4f}",
            "Higher is better"
        )
    
    # Create prediction vs actual plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(
            color='#ff1801',
            size=8
        )
    ))
    
    # Add perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(
            color='#1E88E5',
            dash='dash'
        )
    ))
    
    fig.update_layout(
        title="Predicted vs Actual Lap Times",
        xaxis_title="Actual Lap Time (seconds)",
        yaxis_title="Predicted Lap Time (seconds)",
        template="plotly_white",
        hoverlabel=dict(bgcolor="white"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': features.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    st.subheader("Feature Importance Analysis")
    
    fig_importance = go.Figure()
    
    fig_importance.add_trace(go.Bar(
        x=feature_importance['feature'],
        y=feature_importance['importance'],
        marker_color='#ff1801'
    ))
    
    fig_importance.update_layout(
        title="Feature Importance in Lap Time Prediction",
        xaxis_title="Feature",
        yaxis_title="Importance Score",
        template="plotly_white",
        hoverlabel=dict(bgcolor="white"),
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    st.plotly_chart(fig_importance, use_container_width=True)

def render_performance_trends(lap_data):
    """Render performance trends analysis."""
    st.subheader("Performance Trends Analysis")
    
    # Calculate performance trends
    trends = calculate_performance_trends(lap_data)
    
    # Create trend visualization
    create_trend_visualization(trends)
    
    # Display trend insights
    display_trend_insights(trends)

def calculate_performance_trends(lap_data):
    """Calculate performance trends for each driver."""
    trends = []
    
    for driver in lap_data['driver_number'].unique():
        driver_laps = lap_data[lap_data['driver_number'] == driver]
        
        # Calculate trend metrics
        trend = {
            'driver_number': driver,
            'improvement_rate': calculate_improvement_rate(driver_laps),
            'consistency_trend': calculate_consistency_trend(driver_laps),
            'performance_score': calculate_performance_score(driver_laps)
        }
        
        trends.append(trend)
    
    return pd.DataFrame(trends)

def calculate_improvement_rate(driver_laps):
    """Calculate the rate of improvement in lap times."""
    if len(driver_laps) < 2:
        return 0
    
    # Fit linear regression to lap times
    x = np.arange(len(driver_laps)).reshape(-1, 1)
    y = driver_laps['lap_time_seconds'].values
    
    # Calculate slope using numpy
    slope = np.polyfit(x.flatten(), y, 1)[0]
    
    return -slope  # Negative slope indicates improvement

def calculate_consistency_trend(driver_laps):
    """Calculate the trend in lap time consistency."""
    if len(driver_laps) < 3:
        return 0
    
    # Calculate rolling standard deviation
    rolling_std = driver_laps['lap_time_seconds'].rolling(window=3, min_periods=1).std()
    
    # Calculate trend in consistency
    consistency_trend = -np.polyfit(np.arange(len(rolling_std)), rolling_std, 1)[0]
    
    return consistency_trend

def calculate_performance_score(driver_laps):
    """Calculate overall performance score."""
    if len(driver_laps) < 3:
        return 0
    
    # Combine improvement rate and consistency
    improvement_rate = calculate_improvement_rate(driver_laps)
    consistency_trend = calculate_consistency_trend(driver_laps)
    
    # Normalize and combine scores
    score = (improvement_rate * 0.6) + (consistency_trend * 0.4)
    
    return score

def create_trend_visualization(trends):
    """Create visualization for performance trends."""
    fig = go.Figure()
    
    # Add improvement rate trace
    fig.add_trace(go.Bar(
        name='Improvement Rate',
        x=trends['driver_number'],
        y=trends['improvement_rate'],
        marker_color='#ff1801'
    ))
    
    # Add consistency trend trace
    fig.add_trace(go.Bar(
        name='Consistency Trend',
        x=trends['driver_number'],
        y=trends['consistency_trend'],
        marker_color='#1E88E5'
    ))
    
    fig.update_layout(
        title="Driver Performance Trends",
        xaxis_title="Driver Number",
        yaxis_title="Trend Score",
        barmode='group',
        template="plotly_white",
        hoverlabel=dict(bgcolor="white"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_trend_insights(trends):
    """Display insights from performance trends."""
    # Find top performers
    top_improvers = trends.nlargest(3, 'improvement_rate')
    top_consistent = trends.nlargest(3, 'consistency_trend')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Top Improvers")
        for _, driver in top_improvers.iterrows():
            st.metric(
                f"Driver {int(driver['driver_number'])}",
                f"{driver['improvement_rate']:.3f}s/lap",
                "Improvement Rate"
            )
    
    with col2:
        st.markdown("### Most Consistent")
        for _, driver in top_consistent.iterrows():
            st.metric(
                f"Driver {int(driver['driver_number'])}",
                f"{driver['consistency_trend']:.3f}",
                "Consistency Score"
            )

def render_race_simulation(lap_data):
    """Render race simulation analysis."""
    st.subheader("Race Simulation")
    
    # Prepare simulation parameters
    total_laps = st.slider(
        "Total Laps to Simulate",
        min_value=10,
        max_value=70,
        value=50,
        step=5
    )
    
    # Run simulation
    simulation_results = simulate_race(lap_data, total_laps)
    
    # Display simulation results
    display_simulation_results(simulation_results)

def simulate_race(lap_data, total_laps):
    """Simulate race progression based on historical data."""
    simulation_results = []
    
    for driver in lap_data['driver_number'].unique():
        driver_laps = lap_data[lap_data['driver_number'] == driver]
        
        # Calculate driver parameters
        avg_lap_time = driver_laps['lap_time_seconds'].mean()
        lap_time_std = driver_laps['lap_time_seconds'].std()
        
        # Simulate lap times
        simulated_laps = np.random.normal(
            avg_lap_time,
            lap_time_std,
            total_laps
        )
        
        # Calculate cumulative time
        cumulative_time = np.cumsum(simulated_laps)
        
        simulation_results.append({
            'driver_number': driver,
            'final_time': cumulative_time[-1],
            'avg_lap_time': avg_lap_time,
            'consistency': lap_time_std,
            'lap_times': simulated_laps,
            'cumulative_times': cumulative_time
        })
    
    return pd.DataFrame(simulation_results).sort_values('final_time')

def display_simulation_results(simulation_results):
    """Display race simulation results."""
    # Create race progression chart
    fig = go.Figure()
    
    for _, driver in simulation_results.iterrows():
        fig.add_trace(go.Scatter(
            x=np.arange(len(driver['cumulative_times'])),
            y=driver['cumulative_times'],
            mode='lines',
            name=f"Driver {int(driver['driver_number'])}",
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title="Simulated Race Progression",
        xaxis_title="Lap Number",
        yaxis_title="Cumulative Time (seconds)",
        template="plotly_white",
        hoverlabel=dict(bgcolor="white"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display final standings
    st.markdown("### Predicted Final Standings")
    
    for i, (_, driver) in enumerate(simulation_results.iterrows(), 1):
        gap = (
            f"+{driver['final_time'] - simulation_results.iloc[0]['final_time']:.3f}s"
            if i > 1 else "Leader"
        )
        
        st.metric(
            f"{i}. Driver {int(driver['driver_number'])}",
            f"{driver['final_time']:.3f}s",
            gap
        ) 