"""
Helper functions for the F1 Data Visualization app.
This file contains helper functions to keep the main app.py file cleaner.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from utils import F1DataAPI, process_lap_times, calculate_advanced_statistics

def load_session_data(session_id):
    """Load session data with error handling."""
    with st.spinner("Loading session data..."):
        session_data, error = F1DataAPI.get_session_data(session_id)
        if error:
            st.error(f"Error loading session data: {error}")
            st.stop()
        return session_data

def load_driver_data(session_id):
    """Load driver data with error handling."""
    with st.spinner("Loading driver data..."):
        drivers_df, error = F1DataAPI.get_driver_list(session_id)
        if error:
            st.error(f"Error loading driver data: {error}")
            st.stop()
        if not drivers_df:
            st.warning("No driver data available for this session")
            return None
        return pd.DataFrame(drivers_df)

def load_lap_times(session_id, driver_number=None):
    """Load lap times with error handling."""
    with st.spinner("Loading lap times..."):
        lap_data, error = F1DataAPI.get_lap_times(session_id, driver_number)
        if error:
            st.error(f"Error loading lap times: {error}")
            return None
        if lap_data is None or lap_data.empty:
            return None
        return process_lap_times(lap_data)

def calculate_driver_stats(driver_laps):
    """Calculate statistics for a driver."""
    if driver_laps is None or driver_laps.empty:
        return None
    
    return {
        'avg_lap': driver_laps['lap_time_seconds'].mean(),
        'fastest_lap': driver_laps['lap_time_seconds'].min(),
        'median_lap': driver_laps['lap_time_seconds'].median(),
        'std_dev': driver_laps['lap_time_seconds'].std(),
        'consistency_score': 1 / (driver_laps['lap_time_seconds'].std() / driver_laps['lap_time_seconds'].mean()),
        'lap_count': len(driver_laps)
    }

def create_lap_time_chart(driver1_laps, driver2_laps, driver1_name, driver2_name):
    """Create a lap time comparison chart."""
    fig = go.Figure()
    
    # Add traces for each driver
    fig.add_trace(go.Scatter(
        x=driver1_laps['lap_number'],
        y=driver1_laps['lap_time_seconds'],
        mode='lines+markers',
        name=driver1_name,
        line=dict(color='#1E88E5', width=2),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=driver2_laps['lap_number'],
        y=driver2_laps['lap_time_seconds'],
        mode='lines+markers',
        name=driver2_name,
        line=dict(color='#FFC107', width=2),
        marker=dict(size=8)
    ))
    
    # Add fastest lap markers
    fastest_lap1 = driver1_laps['lap_time_seconds'].min()
    fastest_lap_idx1 = driver1_laps['lap_time_seconds'].idxmin()
    fastest_lap_num1 = driver1_laps.iloc[fastest_lap_idx1]['lap_number']
    
    fastest_lap2 = driver2_laps['lap_time_seconds'].min()
    fastest_lap_idx2 = driver2_laps['lap_time_seconds'].idxmin()
    fastest_lap_num2 = driver2_laps.iloc[fastest_lap_idx2]['lap_number']
    
    fig.add_trace(go.Scatter(
        x=[fastest_lap_num1],
        y=[fastest_lap1],
        mode='markers',
        marker=dict(
            color='#1E88E5',
            size=12,
            symbol='star',
            line=dict(width=2, color='white')
        ),
        name=f"{driver1_name} Fastest",
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=[fastest_lap_num2],
        y=[fastest_lap2],
        mode='markers',
        marker=dict(
            color='#FFC107',
            size=12,
            symbol='star',
            line=dict(width=2, color='white')
        ),
        name=f"{driver2_name} Fastest",
        showlegend=True
    ))
    
    # Update layout
    fig.update_layout(
        title="Lap Time Progression",
        xaxis_title="Lap Number",
        yaxis_title="Lap Time (seconds)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="closest"
    )
    
    return fig

def create_team_stats(lap_data, drivers_df):
    """Calculate team statistics from lap data."""
    if lap_data is None or lap_data.empty or drivers_df is None or drivers_df.empty:
        return None
    
    # Merge lap data with driver info to get team names
    merged_data = pd.merge(
        lap_data, 
        drivers_df[['driver_number', 'team_name']], 
        on='driver_number', 
        how='left'
    )
    
    # Group by team and calculate statistics
    team_stats = []
    
    for team_name, team_data in merged_data.groupby('team_name'):
        team_laps = team_data['lap_time_seconds'].dropna()
        
        if len(team_laps) > 0:
            stats = {
                'team_name': team_name,
                'avg_lap_time': team_laps.mean(),
                'fastest_lap': team_laps.min(),
                'lap_count': len(team_data),
                'std_dev': team_laps.std(),
                'consistency_score': 1 / (team_laps.std() / team_laps.mean()) if team_laps.std() > 0 else 0,
                'driver_count': team_data['driver_number'].nunique()
            }
            team_stats.append(stats)
    
    return pd.DataFrame(team_stats) 