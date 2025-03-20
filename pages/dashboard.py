import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging
from utils import (
    F1DataAPI, 
    format_lap_time, 
    calculate_advanced_statistics,
    process_lap_times
)
from styles import SESSION_CARD_STYLE
from helpers import load_session_data, load_lap_times
import time

# Configure logging
logger = logging.getLogger(__name__)

@st.cache_data(ttl=300)
def get_optimized_session_results(session_id):
    """Get session results with optimized caching."""
    results_df, error = F1DataAPI.get_session_results(session_id)
    return results_df, error

@st.cache_data(ttl=300)
def get_optimized_lap_times(session_id):
    """Get lap times data with optimized caching."""
    lap_times_df, error = F1DataAPI.get_lap_times(session_id)
    return lap_times_df, error

def render_dashboard(session_id, selected_session):
    """Render the dashboard page with optimized visualizations."""
    st.header("Session Overview")
    
    # Fetch session data with error handling and retry mechanism
    with st.spinner("Loading session data..."):
        retry_count = 0
        max_retries = 3
        session_data = None
        error = None
        
        while retry_count < max_retries and not session_data:
            session_data, error = F1DataAPI.get_session_data(session_id)
            if error and "timeout" in str(error).lower():
                retry_count += 1
                with st.status(f"Retrying... ({retry_count}/{max_retries})"):
                    time.sleep(1)
            else:
                break
                
        if error:
            st.error(f"Error loading session data: {error}")
            st.info("Showing limited information due to data loading error.")
            display_fallback_data(session_id)
            return
            
        if session_data:
            # Display session information with enhanced styling
            st.markdown(SESSION_CARD_STYLE, unsafe_allow_html=True)
            
            # Get session info from selected_session if available
            session_type = selected_session.get('session_type', session_data.get('session_type', 'N/A'))
            track_name = selected_session.get('track_name', session_data.get('track_name', 'N/A'))
            date = selected_session.get('date', session_data.get('date', 'N/A'))
            if isinstance(date, str) and 'T' in date:
                date = date.split('T')[0]
            country_name = selected_session.get('country_name', session_data.get('country_name', 'N/A'))
            
            # Advanced session card with more metrics
            st.markdown(f"""
            <div class="session-card">
                <h3>🏁 Session Information</h3>
                <div class="metric-container">
                    <div class="metric-item">
                        <div class="metric-value">{session_type}</div>
                        <div class="metric-label">Session Type</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">{track_name}</div>
                        <div class="metric-label">Track</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">{date}</div>
                        <div class="metric-label">Date</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value">{country_name}</div>
                        <div class="metric-label">Country</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Fetch and display session results with retry mechanism
            with st.spinner("Loading session results..."):
                retry_count = 0
                results_df = None
                
                while retry_count < max_retries and results_df is None:
                    results_df, error = get_optimized_session_results(session_id)
                    if error and "timeout" in str(error).lower():
                        retry_count += 1
                        with st.status(f"Retrying... ({retry_count}/{max_retries})"):
                            time.sleep(1)
                    else:
                        break
                
                if error:
                    st.error(f"Error loading session results: {error}")
                    display_fallback_data(session_id)
                    return
                    
                if results_df is not None and not results_df.empty:
                    # Create tabs with enhanced styling
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📊 Results Table",
                        "📈 Performance Charts",
                        "🏎️ Team Analysis",
                        "🔍 Detailed Stats"
                    ])
                    
                    with tab1:
                        render_results_table(results_df)
                    
                    with tab2:
                        render_performance_charts(results_df, session_id)
                    
                    with tab3:
                        render_team_analysis(results_df, session_id)
                        
                    with tab4:
                        render_detailed_stats(results_df, session_id)
                else:
                    display_fallback_data(session_id)

def render_results_table(results_df):
    """Render an enhanced results table with custom styling."""
    st.subheader("Session Results")
    
    # Format the results for display
    display_df = results_df.copy()
    
    # Format fastest lap time
    if 'fastest_lap' in display_df.columns:
        display_df['fastest_lap'] = display_df['fastest_lap'].apply(
            lambda x: format_lap_time(x) if pd.notnull(x) else "N/A"
        )
    
    # Rename columns for display
    display_df.columns = ['Pos', 'Number', 'Driver', 'Team', 'Fastest Lap', 'Laps']
    
    # Add styling to the dataframe
    st.dataframe(
        display_df,
        hide_index=True,
        column_config={
            "Pos": st.column_config.NumberColumn(
                "Position",
                help="Driver's position based on fastest lap",
                format="%d",
                width="small"
            ),
            "Number": st.column_config.NumberColumn(
                "Number",
                help="Driver's car number",
                width="small"
            ),
            "Driver": st.column_config.TextColumn(
                "Driver",
                help="Driver's name",
                width="medium"
            ),
            "Team": st.column_config.TextColumn(
                "Team",
                help="Driver's team",
                width="medium"
            ),
            "Fastest Lap": st.column_config.TextColumn(
                "Fastest Lap",
                help="Driver's fastest lap time",
                width="medium"
            ),
            "Laps": st.column_config.NumberColumn(
                "Laps",
                help="Number of laps completed",
                width="small"
            )
        }
    )
    
    # Add download button for results
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name=f"session_results.csv",
        mime="text/csv",
    )

def render_performance_charts(results_df, session_id):
    """Render enhanced performance visualization charts."""
    try:
        if results_df is None or results_df.empty:
            st.warning("No results data available")
            return
            
        # Get lap times
        lap_times_df, error = F1DataAPI.get_lap_times(session_id)
        if error:
            st.error(f"Error loading lap times: {error}")
            return
            
        if lap_times_df is None or lap_times_df.empty:
            st.warning("No lap time data available")
            return
            
        # Process lap times to handle NaN values
        processed_laps = process_lap_times(lap_times_df)
        if processed_laps.empty:
            st.warning("No valid lap times available for analysis")
            return
            
        # Calculate driver consistency
        driver_stats = processed_laps.groupby('driver_number').agg({
            'lap_time_seconds': ['std', 'mean']
        }).reset_index()
        
        # Flatten column names
        driver_stats.columns = ['driver_number', 'std_dev', 'avg_time']
        
        # Remove any NaN values
        driver_stats = driver_stats.dropna()
        
        if driver_stats.empty:
            st.warning("No valid consistency data available")
            return
            
        # Get driver names
        driver_info = results_df[['driver_number', 'driver_name']].copy()
        
        # Merge with driver info
        driver_consistency = pd.merge(
            driver_stats,
            driver_info,
            on='driver_number',
            how='left'
        )
        
        # Sort by consistency (lower is better)
        driver_consistency = driver_consistency.sort_values('std_dev')
        
        # Create consistency chart
        fig = go.Figure()
        
        # Add bars for consistency
        fig.add_trace(go.Bar(
            x=driver_consistency['driver_name'],
            y=driver_consistency['std_dev'],
            marker_color='#1E88E5',
            name='Lap Time Consistency',
            text=driver_consistency['std_dev'].apply(lambda x: f"{x:.3f}s"),
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Lap Time Consistency (Standard Deviation)",
            xaxis_title="Driver",
            yaxis_title="Standard Deviation (seconds)",
            template="plotly_white",
            showlegend=False,
            hoverlabel=dict(bgcolor="white"),
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create average lap time chart
        fig = go.Figure()
        
        # Add bars for average lap time
        fig.add_trace(go.Bar(
            x=driver_consistency['driver_name'],
            y=driver_consistency['avg_time'],
            marker_color='#ff1801',
            name='Average Lap Time',
            text=driver_consistency['avg_time'].apply(lambda x: f"{x:.3f}s"),
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Average Lap Times by Driver",
            xaxis_title="Driver",
            yaxis_title="Average Lap Time (seconds)",
            template="plotly_white",
            showlegend=False,
            hoverlabel=dict(bgcolor="white"),
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        logger.error(f"Error in performance charts: {str(e)}")
        st.error("An error occurred while rendering performance charts")

def render_team_analysis(results_df, session_id):
    """Render enhanced team performance analysis."""
    st.subheader("Team Performance")
    
    # Ensure we have the required columns
    if results_df is None or results_df.empty:
        st.warning("No results data available for team analysis")
        return
        
    # Check and rename team column if necessary
    team_column = None
    possible_team_columns = ['Team', 'team_name', 'team']
    for col in possible_team_columns:
        if col in results_df.columns:
            team_column = col
            break
            
    if team_column is None:
        st.warning("Team information not available in the data")
        return
        
    # Create a copy and standardize column names
    analysis_df = results_df.copy()
    analysis_df = analysis_df.rename(columns={team_column: 'Team'})
    
    # Ensure we have the required columns for analysis
    required_columns = ['Team', 'fastest_lap', 'Laps']
    missing_columns = set(required_columns) - set(analysis_df.columns)
    if missing_columns:
        st.warning(f"Missing required columns for team analysis: {missing_columns}")
        return
    
    # Group data by team
    try:
        team_stats = analysis_df.groupby('Team').agg({
            'fastest_lap': ['min', 'mean'],
            'Laps': 'sum'
        }).reset_index()
        
        # Flatten column names
        team_stats.columns = ['Team', 'Best Lap', 'Average Lap', 'Total Laps']
        
        # Remove any NaN values
        team_stats = team_stats.dropna()
        
        if team_stats.empty:
            st.warning("No valid team statistics available")
            return
            
        # Sort by best lap time
        team_stats = team_stats.sort_values('Best Lap')
        
        # Create team performance visualization
        fig = go.Figure()
        
        # Add best lap bars
        fig.add_trace(go.Bar(
            name='Best Lap',
            x=team_stats['Team'],
            y=team_stats['Best Lap'],
            marker_color='#ff1801',
            text=team_stats['Best Lap'].apply(lambda x: f"{x:.3f}s"),
            textposition='auto'
        ))
        
        # Add average lap bars
        fig.add_trace(go.Bar(
            name='Average Lap',
            x=team_stats['Team'],
            y=team_stats['Average Lap'],
            marker_color='#1E88E5',
            text=team_stats['Average Lap'].apply(lambda x: f"{x:.3f}s"),
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Team Lap Time Performance",
            xaxis_title="Team",
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
        
        # Display team metrics table
        st.subheader("Team Metrics")
        team_stats['Delta (Best-Avg)'] = team_stats['Average Lap'] - team_stats['Best Lap']
        
        st.dataframe(
            team_stats,
            hide_index=True,
            column_config={
                "Team": st.column_config.TextColumn(
                    "Team",
                    help="Team name",
                    width="medium"
                ),
                "Best Lap": st.column_config.NumberColumn(
                    "Best Lap",
                    help="Team's best lap time in seconds",
                    format="%.3f s",
                    width="medium"
                ),
                "Average Lap": st.column_config.NumberColumn(
                    "Average Lap",
                    help="Team's average lap time in seconds",
                    format="%.3f s",
                    width="medium"
                ),
                "Total Laps": st.column_config.NumberColumn(
                    "Total Laps",
                    help="Total number of laps completed by the team",
                    format="%d",
                    width="medium"
                ),
                "Delta (Best-Avg)": st.column_config.NumberColumn(
                    "Delta (Best-Avg)",
                    help="Difference between average and best lap time",
                    format="%.3f s",
                    width="medium"
                )
            }
        )
    except Exception as e:
        logger.error(f"Error in team analysis: {str(e)}")
        st.error("An error occurred while analyzing team performance")

def render_detailed_stats(results_df, session_id):
    """Render detailed statistics."""
    try:
        st.subheader("Detailed Statistics")
        
        # Fetch lap times
        lap_times_df, error = get_optimized_lap_times(session_id)
        if error or lap_times_df is None or lap_times_df.empty:
            st.warning("Detailed statistics not available for this session")
            return
        
        # Process lap times
        lap_times_df = pd.DataFrame(lap_times_df)
        
        # Get driver information from results
        driver_info = {}
        for _, row in results_df.iterrows():
            driver_num = row.get('Number', row.get('driver_number'))
            if driver_num:
                driver_name = row.get('Driver', row.get('driver_name', f'Driver {driver_num}'))
                team_name = row.get('Team', row.get('team_name', 'Unknown Team'))
                driver_info[driver_num] = {
                    'name': driver_name,
                    'team': team_name
                }
        
        # Driver selector for detailed analysis
        driver_numbers = lap_times_df['driver_number'].unique()
        driver_options = []
        
        for num in driver_numbers:
            if num in driver_info:
                driver_options.append({
                    'label': f"{driver_info[num]['name']} - {driver_info[num]['team']} ({num})",
                    'value': num
                })
            else:
                driver_options.append({
                    'label': f"Driver {num}",
                    'value': num
                })
        
        if not driver_options:
            st.warning("No driver data available for analysis")
            return
            
        # Sort options by driver name
        driver_options.sort(key=lambda x: x['label'])
        
        # Create selectbox with formatted options
        selected_driver_option = st.selectbox(
            "Select Driver for Detailed Analysis",
            options=[opt['label'] for opt in driver_options],
            format_func=lambda x: x
        )
        
        # Extract driver number from selection
        selected_driver = None
        for opt in driver_options:
            if opt['label'] == selected_driver_option:
                selected_driver = opt['value']
                break
                
        if selected_driver is None:
            st.error("Could not determine selected driver")
            return
        
        # Filter lap times for selected driver
        driver_laps = lap_times_df[lap_times_df['driver_number'] == selected_driver].copy()
        
        if driver_laps.empty:
            st.warning(f"No lap data available for selected driver")
            return
        
        # Convert lap times to numeric, handling any string formats
        driver_laps.loc[:, 'lap_time'] = pd.to_numeric(driver_laps['lap_time'], errors='coerce')
        
        # Remove any NaN values
        driver_laps = driver_laps.dropna(subset=['lap_time'])
        
        if len(driver_laps) < 1:
            st.warning("No valid lap times available for analysis")
            return
            
        # Calculate statistics
        driver_stats = {
            'Total Laps': len(driver_laps),
            'Fastest Lap': driver_laps['lap_time'].min(),
            'Average Lap': driver_laps['lap_time'].mean(),
            'Median Lap': driver_laps['lap_time'].median(),
            'Std Deviation': driver_laps['lap_time'].std(),
        }
        
        # Only calculate consistency score if we have valid standard deviation
        if driver_stats['Std Deviation'] > 0:
            driver_stats['Consistency Score'] = 100 * (1 - (driver_stats['Std Deviation'] / driver_stats['Average Lap']))
        else:
            driver_stats['Consistency Score'] = 100.0
        
        # Display statistics in 2 columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Laps", f"{driver_stats['Total Laps']}")
            st.metric("Fastest Lap", format_lap_time(driver_stats['Fastest Lap']))
            st.metric("Average Lap", format_lap_time(driver_stats['Average Lap']))
        
        with col2:
            st.metric("Median Lap", format_lap_time(driver_stats['Median Lap']))
            st.metric("Std Deviation", f"{driver_stats['Std Deviation']:.3f}s")
            st.metric("Consistency Score", f"{driver_stats['Consistency Score']:.1f}%")
        
        # Create lap time progression chart
        st.subheader(f"Lap Time Progression - {selected_driver_option}")
        
        fig = go.Figure()
        
        # Add lap times
        fig.add_trace(go.Scatter(
            x=driver_laps['lap_number'],
            y=driver_laps['lap_time'],
            mode='lines+markers',
            name='Lap Time',
            line=dict(color='#ff1801', width=2),
            marker=dict(size=8)
        ))
        
        # Add moving average
        window_size = min(5, len(driver_laps))
        if window_size > 1:
            rolling_avg = driver_laps['lap_time'].rolling(window=window_size, min_periods=1).mean()
            
            fig.add_trace(go.Scatter(
                x=driver_laps['lap_number'],
                y=rolling_avg,
                mode='lines',
                name=f'{window_size}-Lap Moving Avg',
                line=dict(color='#1E88E5', width=2, dash='dash')
            ))
        
        # Add fastest lap marker
        fastest_lap = driver_laps['lap_time'].min()
        fastest_lap_idx = driver_laps['lap_time'].idxmin()
        fastest_lap_num = driver_laps.iloc[driver_laps.index.get_indexer([fastest_lap_idx])[0]]['lap_number']
        
        fig.add_trace(go.Scatter(
            x=[fastest_lap_num],
            y=[fastest_lap],
            mode='markers',
            marker=dict(
                color='#00E676',
                size=12,
                symbol='star',
                line=dict(width=2, color='white')
            ),
            name='Fastest Lap',
            showlegend=True
        ))
        
        fig.update_layout(
            title=f"Lap Time Progression",
            xaxis_title="Lap Number",
            yaxis_title="Lap Time (seconds)",
            template="plotly_white",
            hoverlabel=dict(bgcolor="white"),
            hovermode="x unified",
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
        
    except Exception as e:
        logger.error(f"Error in detailed stats: {str(e)}")
        st.error("An error occurred while rendering detailed statistics")

def display_fallback_data(session_id):
    """Display fallback data when results are not available."""
    drivers_df, error = F1DataAPI.get_driver_list(session_id)
    if error:
        st.error(f"Error loading driver data: {error}")
    elif drivers_df and len(drivers_df) > 0:
        drivers_df = pd.DataFrame(drivers_df)
        st.subheader("Participating Drivers")
        st.dataframe(
            drivers_df[['driver_number', 'driver_name', 'team_name']],
            hide_index=True,
            column_config={
                "driver_number": st.column_config.NumberColumn(
                    "Number",
                    help="Driver's car number",
                    width="small"
                ),
                "driver_name": st.column_config.TextColumn(
                    "Driver",
                    help="Driver's name",
                    width="medium"
                ),
                "team_name": st.column_config.TextColumn(
                    "Team",
                    help="Driver's team",
                    width="medium"
                )
            }
        ) 