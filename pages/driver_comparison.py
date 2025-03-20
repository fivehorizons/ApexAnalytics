import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import logging
from utils import (
    F1DataAPI, 
    process_lap_times, 
    calculate_advanced_statistics,
    format_lap_time
)
from styles import DRIVER_COMPARISON_STYLE

# Configure logging
logger = logging.getLogger(__name__)

def render_driver_comparison(session_id: str):
    """Render the driver comparison page."""
    try:
        # First get driver list to get names
        drivers_df, error = F1DataAPI.get_driver_list(session_id)
        if error:
            st.error(f"Error loading driver data: {error}")
            return
            
        if not drivers_df or len(drivers_df) == 0:
            st.warning("No driver data available for this session")
            return
            
        # Convert to DataFrame if it's a list
        if isinstance(drivers_df, list):
            drivers_df = pd.DataFrame(drivers_df)
        
        # Display raw drivers data for debugging
        st.write("Available driver data columns:", list(drivers_df.columns))
        
        # Get lap times
        lap_times_df, error = F1DataAPI.get_lap_times(session_id)
        if error:
            st.error(f"Error loading lap times: {error}")
            return
            
        if lap_times_df is None or lap_times_df.empty:
            st.warning("No lap time data available for this session.")
            return
            
        # Process lap times
        processed_laps = process_lap_times(lap_times_df)
        if processed_laps.empty:
            st.warning("No valid lap times available for comparison.")
            return
            
        # Create driver mapping - try all possible column name variations
        driver_mapping = {}
        for _, row in drivers_df.iterrows():
            # Try different variations of driver number column
            for num_col in ['driver_number', 'Number', 'number', 'DriverNumber']:
                driver_num = row.get(num_col)
                if driver_num is not None:
                    break
            
            # If we found a driver number
            if driver_num is not None:
                # Try different variations of driver name column
                driver_name = None
                for name_col in ['driver_name', 'Driver', 'FullName', 'name', 'Abbreviation', 'code']:
                    if name_col in row and pd.notna(row[name_col]):
                        driver_name = row[name_col]
                        break
                
                # If no name found, use driver number
                if driver_name is None or pd.isna(driver_name):
                    driver_name = f"Driver {driver_num}"
                
                # Try different variations of team name column
                team_name = None
                for team_col in ['team_name', 'Team', 'team', 'Constructor']:
                    if team_col in row and pd.notna(row[team_col]):
                        team_name = row[team_col]
                        break
                
                # If no team found, use unknown
                if team_name is None or pd.isna(team_name):
                    team_name = "Unknown Team"
                
                # Create full display name
                display_name = f"{driver_name} - {team_name} ({driver_num})"
                
                # Store in mapping
                driver_mapping[str(driver_num)] = {
                    'name': driver_name,
                    'team': team_name,
                    'display': display_name
                }
        
        # Log the driver mapping for debugging
        logger.info(f"Driver mapping: {driver_mapping}")
                
        # Get unique drivers with lap times
        available_drivers = []
        for driver_num in processed_laps['driver_number'].unique():
            # Convert to string for consistent lookup
            driver_key = str(driver_num)
            if driver_key in driver_mapping:
                available_drivers.append({
                    'number': driver_num,
                    'display': driver_mapping[driver_key]['display']
                })
            else:
                # Fallback if driver not in mapping
                available_drivers.append({
                    'number': driver_num,
                    'display': f"Driver {driver_num}"
                })
                
        if len(available_drivers) < 2:
            st.warning("Not enough drivers with valid lap times for comparison.")
            return
            
        # Sort drivers by display name
        available_drivers.sort(key=lambda x: x['display'])
        
        # Driver selection
        col1, col2 = st.columns(2)
        with col1:
            if 'driver1' in st.session_state and isinstance(st.session_state.driver1, int) and st.session_state.driver1 < len(available_drivers):
                default_idx1 = st.session_state.driver1
            else:
                default_idx1 = 0
            
            selected_driver1 = st.selectbox(
                "Select first driver",
                options=[d['display'] for d in available_drivers],
                index=default_idx1,
                key="driver1"
            )
            
            # Safely get driver number using a try/except block to handle potential errors
            try:
                driver1_num = next(d['number'] for d in available_drivers if d['display'] == selected_driver1)
            except StopIteration:
                if available_drivers:
                    # Fallback to the first driver if the selected one isn't found
                    st.warning(f"Selected driver '{selected_driver1}' not found. Using first available driver instead.")
                    driver1_num = available_drivers[0]['number']
                    selected_driver1 = available_drivers[0]['display']
                else:
                    st.error("No drivers available for selection.")
                    return
            
        with col2:
            # Remove first driver from options
            remaining_drivers = [d for d in available_drivers if d['display'] != selected_driver1]
            
            if not remaining_drivers:
                st.error("No additional drivers available for comparison.")
                return
                
            if 'driver2' in st.session_state and isinstance(st.session_state.driver2, int) and st.session_state.driver2 < len(remaining_drivers):
                default_idx2 = st.session_state.driver2
            else:
                default_idx2 = 0
                
            selected_driver2 = st.selectbox(
                "Select second driver",
                options=[d['display'] for d in remaining_drivers],
                index=default_idx2,
                key="driver2"
            )
            
            # Safely get driver number using a try/except block to handle potential errors
            try:
                driver2_num = next(d['number'] for d in available_drivers if d['display'] == selected_driver2)
            except StopIteration:
                if remaining_drivers:
                    # Fallback to the first remaining driver if the selected one isn't found
                    st.warning(f"Selected driver '{selected_driver2}' not found. Using first available driver instead.")
                    driver2_num = remaining_drivers[0]['number']
                    selected_driver2 = remaining_drivers[0]['display']
                else:
                    st.error("No second driver available for comparison.")
                    return
            
        # Get data for selected drivers
        driver1_data = processed_laps[processed_laps['driver_number'] == driver1_num].copy()
        driver2_data = processed_laps[processed_laps['driver_number'] == driver2_num].copy()
        
        if driver1_data.empty or driver2_data.empty:
            st.warning("No valid lap time data for one or both selected drivers.")
            return
            
        # Calculate statistics for each driver
        driver1_stats = calculate_advanced_statistics(driver1_data)
        driver2_stats = calculate_advanced_statistics(driver2_data)
        
        if not driver1_stats or not driver2_stats:
            st.warning("Could not calculate statistics for one or both drivers.")
            return
            
        # Get driver names for display
        driver1_key = str(driver1_num)
        driver2_key = str(driver2_num)
        
        # Get display names (full or fallback)
        driver1_name = driver_mapping.get(driver1_key, {}).get('name', f"Driver {driver1_num}")
        driver2_name = driver_mapping.get(driver2_key, {}).get('name', f"Driver {driver2_num}")
            
        # Display comparison metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(driver1_name)
            st.metric("Best Lap", format_lap_time(driver1_stats.get('fastest_lap', 0)))
            st.metric("Average Lap", format_lap_time(driver1_stats.get('average_lap', 0)))
            st.metric("Consistency", f"{driver1_stats.get('std_dev', 0):.3f}s")
            st.metric("Total Laps", driver1_stats.get('total_laps', 0))
            
        with col2:
            st.subheader(driver2_name)
            st.metric("Best Lap", format_lap_time(driver2_stats.get('fastest_lap', 0)))
            st.metric("Average Lap", format_lap_time(driver2_stats.get('average_lap', 0)))
            st.metric("Consistency", f"{driver2_stats.get('std_dev', 0):.3f}s")
            st.metric("Total Laps", driver2_stats.get('total_laps', 0))
            
        # Create lap time comparison plot
        st.subheader("Lap Time Comparison")
        fig = go.Figure()
        
        # Add traces for each driver
        fig.add_trace(go.Scatter(
            x=driver1_data['lap_number'],
            y=driver1_data['lap_time_seconds'],
            name=driver1_name,
            mode='lines+markers',
            line=dict(color='#ff1801', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=driver2_data['lap_number'],
            y=driver2_data['lap_time_seconds'],
            name=driver2_name,
            mode='lines+markers',
            line=dict(color='#1E88E5', width=2)
        ))
        
        # Update layout
        fig.update_layout(
            title="Lap Time Progression",
            xaxis_title="Lap Number",
            yaxis_title="Lap Time (seconds)",
            template="plotly_white",
            hoverlabel=dict(bgcolor="white"),
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display additional statistics if available
        if len(driver1_data) >= 3 and len(driver2_data) >= 3:
            st.subheader("Advanced Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Skewness", f"{driver1_stats.get('skewness', 0):.3f}")
                st.metric("Kurtosis", f"{driver1_stats.get('kurtosis', 0):.3f}")
                
            with col2:
                st.metric("Skewness", f"{driver2_stats.get('skewness', 0):.3f}")
                st.metric("Kurtosis", f"{driver2_stats.get('kurtosis', 0):.3f}")
                
            # Add lap time delta analysis
            st.subheader("Lap Time Delta")
            merged_laps = pd.merge(
                driver1_data[['lap_number', 'lap_time_seconds']],
                driver2_data[['lap_number', 'lap_time_seconds']],
                on='lap_number',
                suffixes=('_1', '_2')
            )
            
            if not merged_laps.empty:
                fig_delta = go.Figure()
                
                # Calculate delta
                merged_laps['delta'] = merged_laps['lap_time_seconds_1'] - merged_laps['lap_time_seconds_2']
                
                fig_delta.add_trace(go.Bar(
                    x=merged_laps['lap_number'],
                    y=merged_laps['delta'],
                    name='Delta',
                    marker_color=['#ff1801' if x > 0 else '#1E88E5' for x in merged_laps['delta']],
                    hovertemplate="Lap %{x}<br>Delta: %{y:.3f}s<extra></extra>"
                ))
                
                fig_delta.update_layout(
                    title=f"Lap Time Delta ({driver1_name} vs {driver2_name})",
                    xaxis_title="Lap Number",
                    yaxis_title="Time Delta (seconds)",
                    template="plotly_white",
                    hoverlabel=dict(bgcolor="white"),
                    showlegend=False
                )
                
                # Add zero line
                fig_delta.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="gray",
                    opacity=0.5
                )
                
                st.plotly_chart(fig_delta, use_container_width=True)
                
    except Exception as e:
        logger.error(f"Error in driver comparison: {str(e)}", exc_info=True)
        st.error("An error occurred while rendering driver comparison")

def load_driver_laps(session_id, driver_number):
    """Load and process lap times for a specific driver."""
    lap_times_df, error = F1DataAPI.get_lap_times(session_id)
    if error:
        st.error(f"Error loading lap times: {error}")
        return None
        
    if lap_times_df is not None and not lap_times_df.empty:
        processed_laps = process_lap_times(lap_times_df)
        return processed_laps[processed_laps['driver_number'] == driver_number]
    return None

def render_lap_time_comparison(driver1_laps, driver2_laps, driver1_name, driver2_name):
    """Render enhanced lap time comparison visualization."""
    st.subheader("Lap Time Comparison")
    
    fig = go.Figure()
    
    # Add driver 1 lap times
    fig.add_trace(go.Scatter(
        x=driver1_laps['lap_number'],
        y=driver1_laps['lap_time_seconds'],
        mode='lines+markers',
        name=driver1_name,
        line=dict(color='#ff1801', width=2),
        marker=dict(size=6)
    ))
    
    # Add driver 2 lap times
    fig.add_trace(go.Scatter(
        x=driver2_laps['lap_number'],
        y=driver2_laps['lap_time_seconds'],
        mode='lines+markers',
        name=driver2_name,
        line=dict(color='#1E88E5', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Lap Time Progression Comparison",
        xaxis_title="Lap Number",
        yaxis_title="Lap Time (seconds)",
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
    
    # Add lap time difference analysis
    render_lap_time_difference(driver1_laps, driver2_laps, driver1_name, driver2_name)

def render_lap_time_difference(driver1_laps, driver2_laps, driver1_name, driver2_name):
    """Render lap time difference analysis."""
    # Merge lap times on lap number
    merged_laps = pd.merge(
        driver1_laps[['lap_number', 'lap_time_seconds']],
        driver2_laps[['lap_number', 'lap_time_seconds']],
        on='lap_number',
        how='inner',
        suffixes=('_driver1', '_driver2')
    )
    
    # Calculate time difference
    merged_laps['time_diff'] = merged_laps['lap_time_seconds_driver1'] - merged_laps['lap_time_seconds_driver2']
    
    fig = go.Figure()
    
    # Add time difference bars
    fig.add_trace(go.Bar(
        x=merged_laps['lap_number'],
        y=merged_laps['time_diff'],
        marker_color=['#ff1801' if x > 0 else '#1E88E5' for x in merged_laps['time_diff']],
        name='Time Difference'
    ))
    
    fig.update_layout(
        title=f"Lap Time Difference ({driver1_name} vs {driver2_name})",
        xaxis_title="Lap Number",
        yaxis_title="Time Difference (seconds)",
        template="plotly_white",
        showlegend=False,
        hoverlabel=dict(bgcolor="white"),
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    # Add zero line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Equal Times",
        annotation_position="right"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_performance_stats(driver1_laps, driver2_laps, driver1_name, driver2_name):
    """Render enhanced performance statistics comparison."""
    st.subheader("Performance Statistics")
    
    # Calculate statistics for both drivers
    stats1 = calculate_advanced_statistics(driver1_laps) if driver1_laps is not None and not driver1_laps.empty else {}
    stats2 = calculate_advanced_statistics(driver2_laps) if driver2_laps is not None and not driver2_laps.empty else {}
    
    # Create comparison metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### {driver1_name}")
        st.metric("Best Lap", f"{stats1.get('fastest_lap', 0):.3f}s")
        st.metric("Average Lap", f"{stats1.get('average_lap', 0):.3f}s")
        st.metric("Consistency", f"{stats1.get('std_dev', 0):.3f}s")
    
    with col2:
        st.markdown(f"### {driver2_name}")
        st.metric("Best Lap", f"{stats2.get('fastest_lap', 0):.3f}s")
        st.metric("Average Lap", f"{stats2.get('average_lap', 0):.3f}s")
        st.metric("Consistency", f"{stats2.get('std_dev', 0):.3f}s")
    
    # Only create comparison chart if we have data for both drivers
    if stats1 and stats2:
        create_performance_comparison_chart(stats1, stats2, driver1_name, driver2_name)
    else:
        st.warning("Insufficient data to create performance comparison chart")

def create_performance_comparison_chart(stats1, stats2, driver1_name, driver2_name):
    """Create an enhanced performance comparison visualization."""
    metrics = ['fastest_lap', 'average_lap', 'std_dev']
    metric_names = ['Best Lap', 'Average Lap', 'Consistency']
    
    fig = go.Figure()
    
    # Add bars for driver 1
    fig.add_trace(go.Bar(
        name=driver1_name,
        x=metric_names,
        y=[stats1[m] for m in metrics],
        marker_color='#ff1801'
    ))
    
    # Add bars for driver 2
    fig.add_trace(go.Bar(
        name=driver2_name,
        x=metric_names,
        y=[stats2[m] for m in metrics],
        marker_color='#1E88E5'
    ))
    
    fig.update_layout(
        title="Performance Metrics Comparison",
        xaxis_title="Metric",
        yaxis_title="Time (seconds)",
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

def render_detailed_analysis(driver1_laps, driver2_laps, driver1_name, driver2_name):
    """Render detailed performance analysis."""
    st.subheader("Detailed Analysis")
    
    # Create lap time distribution comparison
    fig = go.Figure()
    
    # Add violin plot for driver 1
    fig.add_trace(go.Violin(
        y=driver1_laps['lap_time_seconds'],
        name=driver1_name,
        side='negative',
        line_color='#ff1801',
        fillcolor='rgba(255, 24, 1, 0.3)'
    ))
    
    # Add violin plot for driver 2
    fig.add_trace(go.Violin(
        y=driver2_laps['lap_time_seconds'],
        name=driver2_name,
        side='positive',
        line_color='#1E88E5',
        fillcolor='rgba(30, 136, 229, 0.3)'
    ))
    
    fig.update_layout(
        title="Lap Time Distribution Comparison",
        yaxis_title="Lap Time (seconds)",
        template="plotly_white",
        violingap=0,
        violinmode='overlay',
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
    
    # Add sector time comparison if available
    if 'sector1_time' in driver1_laps.columns and 'sector1_time' in driver2_laps.columns:
        render_sector_comparison(driver1_laps, driver2_laps, driver1_name, driver2_name)

def render_sector_comparison(driver1_laps, driver2_laps, driver1_name, driver2_name):
    """Render sector time comparison analysis."""
    st.subheader("Sector Performance")
    
    # Calculate average sector times
    driver1_sectors = {
        'Sector 1': driver1_laps['sector1_time'].mean(),
        'Sector 2': driver1_laps['sector2_time'].mean(),
        'Sector 3': driver1_laps['sector3_time'].mean()
    }
    
    driver2_sectors = {
        'Sector 1': driver2_laps['sector1_time'].mean(),
        'Sector 2': driver2_laps['sector2_time'].mean(),
        'Sector 3': driver2_laps['sector3_time'].mean()
    }
    
    fig = go.Figure()
    
    # Add radar chart for both drivers
    fig.add_trace(go.Scatterpolar(
        r=[driver1_sectors[s] for s in driver1_sectors.keys()],
        theta=list(driver1_sectors.keys()),
        name=driver1_name,
        fill='toself',
        line_color='#ff1801'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[driver2_sectors[s] for s in driver2_sectors.keys()],
        theta=list(driver2_sectors.keys()),
        name=driver2_name,
        fill='toself',
        line_color='#1E88E5'
    ))
    
    fig.update_layout(
        title="Average Sector Time Comparison",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(driver1_sectors.values()), max(driver2_sectors.values())) * 1.1]
            )
        ),
        showlegend=True,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True) 