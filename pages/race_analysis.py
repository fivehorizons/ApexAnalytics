import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging
import matplotlib.pyplot as plt
from utils import (
    F1DataAPI, 
    process_lap_times, 
    calculate_advanced_statistics, 
    analyze_sector_performance,
    export_data,
    format_lap_time,
    create_sector_heatmap,
    create_interactive_sector_chart,
    visualize_sector_comparison
)
from styles import STATS_GRID_STYLE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def render_race_analysis(session_id: str):
    """Render the race analysis page."""
    try:
        # Convert session_id to integer
        try:
            session_id = int(session_id)
        except (ValueError, TypeError) as e:
            st.error(f"Invalid session ID: {session_id}")
            return
            
        st.title("Race Analysis")
        
        # Create tabs for different analysis views
        tabs = st.tabs([
            "Overview", 
            "Lap Analysis", 
            "Sector Analysis",
            "Intervals",
            "Position Data",
            "Pit Stops",
            "Team Radio",
            "Race Control",
            "Export"
        ])
        
        # Get all necessary data upfront
        with st.spinner("Loading lap times..."):
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
                st.warning("No valid lap times available for analysis.")
                return
        
        # Get sector times
        with st.spinner("Loading sector data..."):
            sector_data, sector_error = F1DataAPI.get_sector_times(session_id)
            if sector_error:
                st.warning(f"Error loading sector data: {sector_error}")
                sector_data = None
        
        # Get intervals data
        with st.spinner("Loading interval data..."):
            intervals_data, intervals_error = F1DataAPI.get_intervals(session_id)
            if intervals_error:
                st.warning(f"Error loading interval data: {intervals_error}")
                intervals_data = None
        
        # Get position data
        with st.spinner("Loading position data..."):
            position_data, position_error = F1DataAPI.get_position_data(session_id)
            if position_error:
                st.warning(f"Error loading position data: {position_error}")
                position_data = None
        
        # Get pit stop data
        with st.spinner("Loading pit stop data..."):
            pit_data, pit_error = F1DataAPI.get_pit_data(session_id)
            if pit_error:
                st.warning(f"Error loading pit stop data: {pit_error}")
                pit_data = None
        
        # Get team radio data
        with st.spinner("Loading team radio data..."):
            radio_data, radio_error = F1DataAPI.get_team_radio(session_id)
            if radio_error:
                st.warning(f"Error loading team radio data: {radio_error}")
                radio_data = None
        
        # Get race control messages
        with st.spinner("Loading race control data..."):
            control_data, control_error = F1DataAPI.get_race_control_messages(session_id)
            if control_error:
                st.warning(f"Error loading race control data: {control_error}")
                control_data = None
        
        # Overview Tab
        with tabs[0]:
            render_overview_tab(processed_laps)
            
        # Lap Analysis Tab
        with tabs[1]:
            render_lap_analysis_tab(processed_laps, session_id)
            
        # Sector Analysis Tab
        with tabs[2]:
            render_sector_analysis_tab(sector_data, session_id)
            
        # Intervals Tab
        with tabs[3]:
            render_intervals_tab(intervals_data, session_id)
            
        # Position Data Tab
        with tabs[4]:
            render_position_tab(position_data, session_id)
            
        # Pit Stops Tab
        with tabs[5]:
            render_pit_stops_tab(pit_data, session_id)
            
        # Team Radio Tab
        with tabs[6]:
            render_team_radio_tab(radio_data, session_id)
            
        # Race Control Tab
        with tabs[7]:
            render_race_control_tab(control_data)
            
        # Export Tab
        with tabs[8]:
            render_export_tab(processed_laps)
            
    except Exception as e:
        st.error(f"An error occurred while rendering race analysis: {str(e)}")
        logger.error(f"Race analysis error: {str(e)}", exc_info=True)

def render_overview_tab(processed_laps):
    """Render the overview tab with race statistics."""
    st.header("Race Overview")
    
    # Calculate race statistics
    race_stats = calculate_advanced_statistics(processed_laps)
    if not race_stats:
        st.warning("Could not calculate race statistics.")
        return
    
    # Display overall race statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Fastest Lap", format_lap_time(race_stats.get('fastest_lap', 0)))
        st.metric("Total Laps", race_stats.get('total_laps', 0))
        
    with col2:
        st.metric("Average Lap", format_lap_time(race_stats.get('average_lap', 0)))
        st.metric("Drivers", race_stats.get('drivers_count', 0))
        
    with col3:
        st.metric("Lap Time Variation", f"{race_stats.get('std_dev', 0):.3f}s")
        if 'consistency_score' in race_stats:
            st.metric("Race Consistency", f"{race_stats.get('consistency_score', 0):.2f}")
    
    # Create lap time distribution plot
    st.subheader("Lap Time Distribution")
    fig = go.Figure()
    
    # Add histogram of lap times
    fig.add_trace(go.Histogram(
        x=processed_laps['lap_time_seconds'],
        nbinsx=30,
        name="Lap Times"
    ))
    
    # Add vertical lines for key statistics
    if 'fastest_lap' in race_stats:
        fig.add_vline(
            x=race_stats['fastest_lap'],
            line_dash="dash",
            line_color="green",
            annotation_text="Fastest Lap"
        )
        
    if 'average_lap' in race_stats:
        fig.add_vline(
            x=race_stats['average_lap'],
            line_dash="dash",
            line_color="red",
            annotation_text="Average Lap"
        )
        
    fig.update_layout(
        title="Distribution of Lap Times",
        xaxis_title="Lap Time (seconds)",
        yaxis_title="Count",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display advanced statistics if available
    if len(processed_laps) >= 3:
        st.subheader("Advanced Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            if 'skewness' in race_stats:
                st.metric("Skewness", f"{race_stats['skewness']:.3f}")
            if 'percentile_5' in race_stats:
                st.metric("5th Percentile", format_lap_time(race_stats['percentile_5']))
                
        with col2:
            if 'kurtosis' in race_stats:
                st.metric("Kurtosis", f"{race_stats['kurtosis']:.3f}")
            if 'percentile_95' in race_stats:
                st.metric("95th Percentile", format_lap_time(race_stats['percentile_95']))

def render_lap_analysis_tab(processed_laps, session_id):
    """Render the lap analysis tab with enhanced visualizations."""
    st.subheader("Lap Time Progression")
    
    # Get driver list for selection
    drivers_df, error = F1DataAPI.get_driver_list(session_id)
    if error:
        st.error(f"Error loading driver data: {error}")
        return
        
    if drivers_df and len(drivers_df) > 0:
        # Convert list of dictionaries to DataFrame if needed
        if isinstance(drivers_df, list):
            drivers_df = pd.DataFrame(drivers_df)
        
        # Create multiselect for drivers
        default_drivers = drivers_df['driver_number'].head(3).tolist()
        selected_drivers = st.multiselect(
            "Select Drivers to Compare",
            options=drivers_df['driver_number'].tolist(),
            default=default_drivers,
            format_func=lambda x: f"{drivers_df[drivers_df['driver_number'] == x]['driver_name'].iloc[0]} ({x})"
        )
        
        if selected_drivers:
            create_lap_progression_chart(processed_laps, selected_drivers, drivers_df)

def create_lap_progression_chart(processed_laps, selected_drivers, drivers_df):
    """Create an enhanced lap progression visualization."""
    fig = go.Figure()
    
    for driver in selected_drivers:
        driver_data = processed_laps[processed_laps['driver_number'] == driver]
        driver_name = drivers_df[drivers_df['driver_number'] == driver]['driver_name'].iloc[0]
        
        fig.add_trace(go.Scatter(
            x=driver_data['lap_number'],
            y=driver_data['lap_time_seconds'],
            mode='lines+markers',
            name=driver_name,
            line=dict(width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title="Lap Time Progression",
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

def render_sector_analysis_tab(sector_data, session_id):
    """Render enhanced sector analysis with new visualizations."""
    if sector_data is None or sector_data.empty:
        st.warning("No sector data available for this session.")
        return
        
    # Get driver list for filtering
    drivers_df, error = F1DataAPI.get_driver_list(session_id)
    if error:
        st.error(f"Error loading driver data: {error}")
        return
        
    # Driver selection
    if isinstance(drivers_df, list):
        drivers_df = pd.DataFrame(drivers_df)
        
    selected_driver = st.selectbox(
        "Select Driver for Sector Analysis",
        options=[None] + drivers_df['driver_number'].tolist(),
        format_func=lambda x: "All Drivers" if x is None else f"{drivers_df[drivers_df['driver_number'] == x]['driver_name'].iloc[0]} ({x})"
    )
    
    # Analyze sector performance
    sector_stats = analyze_sector_performance(sector_data, selected_driver)
    
    # Display sector statistics
    st.subheader("Sector Performance Statistics")
    col1, col2, col3 = st.columns(3)
    
    for i, sector in enumerate(['sector1', 'sector2', 'sector3'], 1):
        col = [col1, col2, col3][i-1]
        with col:
            st.metric(f"Sector {i} Best", f"{sector_stats.get(f'{sector}_min', 0):.3f}s")
            st.metric(f"Sector {i} Average", f"{sector_stats.get(f'{sector}_avg', 0):.3f}s")
            if f'{sector}_trend' in sector_stats:
                st.metric(f"Sector {i} Trend", sector_stats[f'{sector}_trend'])
    
    # Display theoretical best lap if available
    if 'theoretical_best_lap' in sector_stats:
        st.metric(
            "Theoretical Best Lap",
            f"{sector_stats['theoretical_best_lap']:.3f}s",
            delta=f"{-sector_stats['theoretical_vs_actual']:.3f}s vs actual"
        )
    
    # Create sector visualizations
    st.subheader("Sector Time Analysis")
    
    # Add visualization type selector
    viz_type = st.radio(
        "Select Visualization",
        ["Sector Comparison", "Sector Heatmap", "Interactive Timeline"],
        horizontal=True
    )
    
    if viz_type == "Sector Comparison":
        fig = visualize_sector_comparison(sector_stats)
        st.pyplot(fig)
        
    elif viz_type == "Sector Heatmap":
        fig = create_sector_heatmap(sector_data, selected_driver)
        st.pyplot(fig)
        
    else:  # Interactive Timeline
        fig = create_interactive_sector_chart(sector_data, selected_driver)
        st.plotly_chart(fig, use_container_width=True)

def render_intervals_tab(intervals_data, session_id):
    """Render intervals analysis."""
    if intervals_data is None or intervals_data.empty:
        st.warning("No interval data available for this session.")
        return
        
    st.subheader("Race Intervals Analysis")
    
    # Get driver list for reference
    drivers_df, _ = F1DataAPI.get_driver_list(session_id)
    if isinstance(drivers_df, list):
        drivers_df = pd.DataFrame(drivers_df)
    
    # Check available columns and find the interval column
    interval_columns = [col for col in intervals_data.columns if 'interval' in col.lower()]
    if not interval_columns:
        st.warning("No interval data found in the dataset.")
        st.write("Available columns:", intervals_data.columns.tolist())
        return
        
    # Use the first found interval column
    interval_column = interval_columns[0]
    
    # Create interval visualization
    fig = go.Figure()
    
    for driver in intervals_data['driver_number'].unique():
        try:
            driver_data = intervals_data[intervals_data['driver_number'] == driver]
            
            # Get driver name from reference data
            driver_info = drivers_df[drivers_df['driver_number'] == driver]
            driver_name = f"Driver {driver}"  # Default name
            if not driver_info.empty:
                driver_name = driver_info['driver_name'].iloc[0]
            
            fig.add_trace(go.Scatter(
                x=driver_data['date'],
                y=driver_data[interval_column],
                name=f"{driver_name} ({driver})",
                mode='lines+markers'
            ))
        except Exception as e:
            st.warning(f"Error plotting data for driver {driver}: {str(e)}")
            continue
    
    fig.update_layout(
        title=f"Intervals Over Time (using {interval_column})",
        xaxis_title="Time",
        yaxis_title="Interval (seconds)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_position_tab(position_data, session_id):
    """Render position analysis."""
    if position_data is None or position_data.empty:
        st.warning("No position data available for this session.")
        return
        
    st.subheader("Race Position Analysis")
    
    # Get driver list for reference
    drivers_df, _ = F1DataAPI.get_driver_list(session_id)
    if isinstance(drivers_df, list):
        drivers_df = pd.DataFrame(drivers_df)
    
    # Create position visualization
    fig = go.Figure()
    
    for driver in position_data['driver_number'].unique():
        driver_data = position_data[position_data['driver_number'] == driver]
        driver_name = drivers_df[drivers_df['driver_number'] == driver]['driver_name'].iloc[0]
        
        fig.add_trace(go.Scatter(
            x=driver_data['date'],
            y=driver_data['position'],
            name=f"{driver_name} ({driver})",
            mode='lines+markers'
        ))
    
    # Invert y-axis since position 1 is best
    fig.update_layout(
        title="Position Changes Over Time",
        xaxis_title="Time",
        yaxis_title="Position",
        yaxis_autorange='reversed',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_pit_stops_tab(pit_data, session_id):
    """Render pit stop analysis."""
    if pit_data is None or pit_data.empty:
        st.warning("No pit stop data available for this session.")
        return
        
    st.subheader("Pit Stop Analysis")
    
    # Get driver list for reference
    drivers_df, _ = F1DataAPI.get_driver_list(session_id)
    if isinstance(drivers_df, list):
        drivers_df = pd.DataFrame(drivers_df)
    
    # Create pit stop visualization
    fig = go.Figure()
    
    for driver in pit_data['driver_number'].unique():
        driver_data = pit_data[pit_data['driver_number'] == driver]
        driver_name = drivers_df[drivers_df['driver_number'] == driver]['driver_name'].iloc[0]
        
        fig.add_trace(go.Bar(
            x=[driver_name],
            y=[len(driver_data)],
            name=f"Driver {driver}",
            text=[len(driver_data)],
            textposition='auto',
        ))
    
    fig.update_layout(
        title="Number of Pit Stops by Driver",
        xaxis_title="Driver",
        yaxis_title="Number of Pit Stops",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display detailed pit stop data
    st.subheader("Pit Stop Details")
    st.dataframe(
        pit_data.merge(
            drivers_df[['driver_number', 'driver_name']], 
            on='driver_number'
        ),
        use_container_width=True
    )

def render_team_radio_tab(radio_data, session_id):
    """Render team radio analysis."""
    if radio_data is None or radio_data.empty:
        st.warning("No team radio data available for this session.")
        return
        
    st.subheader("Team Radio Communications")
    
    # Get driver list for filtering
    drivers_df, _ = F1DataAPI.get_driver_list(session_id)
    if isinstance(drivers_df, list):
        drivers_df = pd.DataFrame(drivers_df)
    
    # Driver selection for filtering
    selected_driver = st.selectbox(
        "Filter by Driver",
        options=[None] + drivers_df['driver_number'].tolist(),
        format_func=lambda x: "All Drivers" if x is None else f"{drivers_df[drivers_df['driver_number'] == x]['driver_name'].iloc[0]} ({x})"
    )
    
    # Filter data if driver selected
    if selected_driver:
        radio_data = radio_data[radio_data['driver_number'] == selected_driver]
    
    # Display radio messages
    for _, message in radio_data.iterrows():
        with st.expander(
            f"{message['date']} - Driver {message['driver_number']}"
        ):
            if 'recording_url' in message:
                st.audio(message['recording_url'])
            st.write(message.get('message', 'No transcript available'))

def render_race_control_tab(control_data):
    """Render race control messages."""
    if control_data is None or control_data.empty:
        st.warning("No race control messages available for this session.")
        return
        
    st.subheader("Race Control Messages")
    
    # Create a timeline of race control messages
    for _, message in control_data.iterrows():
        with st.expander(
            f"{message['date']} - {message.get('category', 'General')}"
        ):
            st.write(message.get('message', ''))
            if 'flag' in message:
                st.info(f"Flag Status: {message['flag']}")

def render_export_tab(processed_laps):
    """Render the data export tab with enhanced functionality."""
    st.subheader("Export Data")
    
    # Add export buttons in a grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = export_data(processed_laps, 'csv')
        st.download_button(
            "📥 Download CSV",
            csv,
            "lap_times.csv",
            "text/csv",
            key='download-csv',
            help="Download data in CSV format"
        )
    
    with col2:
        excel = export_data(processed_laps, 'excel')
        st.download_button(
            "📊 Download Excel",
            excel,
            "lap_times.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key='download-excel',
            help="Download data in Excel format"
        )
    
    with col3:
        json = export_data(processed_laps, 'json')
        st.download_button(
            "🔄 Download JSON",
            json,
            "lap_times.json",
            "application/json",
            key='download-json',
            help="Download data in JSON format"
        )
    
    # Preview data with enhanced styling
    st.subheader("Data Preview")
    st.dataframe(
        processed_laps.head(10),
        hide_index=True,
        use_container_width=True
    ) 