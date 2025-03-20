import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging
from utils import (
    F1DataAPI, 
    process_lap_times, 
    calculate_advanced_statistics, 
    analyze_sector_performance,
    export_data,
    format_lap_time
)
from styles import STATS_GRID_STYLE

# Configure logging
logger = logging.getLogger(__name__)

def render_race_analysis(session_id: str):
    """Render the race analysis page."""
    try:
        # Get lap times data
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
            
        # Calculate race statistics
        race_stats = calculate_advanced_statistics(processed_laps)
        if not race_stats:
            st.warning("Could not calculate race statistics.")
            return
            
        # Display overall race statistics
        st.subheader("Race Overview")
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
        
        # Create lap time progression plot
        st.subheader("Lap Time Progression")
        fig = go.Figure()
        
        # Group by driver and plot each driver's lap times
        for driver in processed_laps['driver_number'].unique():
            driver_laps = processed_laps[processed_laps['driver_number'] == driver]
            fig.add_trace(go.Scatter(
                x=driver_laps['lap_number'],
                y=driver_laps['lap_time_seconds'],
                name=f"Driver {driver}",
                mode='lines+markers'
            ))
            
        fig.update_layout(
            title="Lap Times Throughout the Race",
            xaxis_title="Lap Number",
            yaxis_title="Lap Time (seconds)",
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
                    
    except Exception as e:
        st.error(f"An error occurred while rendering race analysis: {str(e)}")
        logger.error(f"Race analysis error: {str(e)}", exc_info=True)

def render_overview_tab(processed_laps):
    """Render the overview tab with enhanced statistics visualization."""
    st.subheader("Session Statistics")
    
    # Calculate advanced statistics
    advanced_stats = calculate_advanced_statistics(processed_laps)
    
    # Display statistics in a visually appealing grid
    st.markdown(STATS_GRID_STYLE, unsafe_allow_html=True)
    
    # Create metrics grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Average Lap Time",
            f"{advanced_stats.get('average_lap', 0):.3f}s",
            delta=None
        )
        st.metric(
            "Fastest Lap",
            f"{advanced_stats.get('fastest_lap', 0):.3f}s",
            delta=None
        )
    
    with col2:
        st.metric(
            "Median Lap Time",
            f"{advanced_stats.get('median_lap', 0):.3f}s",
            delta=None
        )
        st.metric(
            "Lap Time Std Dev",
            f"{advanced_stats.get('std_dev', 0):.3f}s",
            delta=None
        )
    
    with col3:
        st.metric(
            "Total Laps",
            str(advanced_stats.get('total_laps', 0)),
            delta=None
        )
        st.metric(
            "Drivers",
            str(advanced_stats.get('drivers_count', 0)),
            delta=None
        )
    
    # Create lap time distribution visualization
    create_lap_time_distribution(processed_laps, advanced_stats)

def create_lap_time_distribution(processed_laps, advanced_stats):
    """Create an enhanced lap time distribution visualization."""
    st.subheader("Lap Time Distribution")
    
    # Create violin plot
    fig = go.Figure()
    
    fig.add_trace(go.Violin(
        y=processed_laps['lap_time_seconds'],
        box_visible=True,
        line_color='#ff1801',
        fillcolor='rgba(255, 24, 1, 0.3)',
        opacity=0.6,
        meanline_visible=True,
        name="Lap Times"
    ))
    
    # Add mean and median lines
    fig.add_hline(
        y=advanced_stats['average_lap'],
        line_dash="dash",
        line_color="red",
        annotation_text="Mean",
        annotation_position="right"
    )
    
    fig.add_hline(
        y=advanced_stats['median_lap'],
        line_dash="dash",
        line_color="green",
        annotation_text="Median",
        annotation_position="left"
    )
    
    fig.update_layout(
        title="Lap Time Distribution Analysis",
        yaxis_title="Lap Time (seconds)",
        template="plotly_white",
        showlegend=False,
        hoverlabel=dict(bgcolor="white"),
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)

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

def render_sector_analysis_tab(processed_laps, session_id):
    """Render the sector analysis tab with enhanced visualizations."""
    st.subheader("Sector Analysis")
    
    # Analyze sector performance
    sector_stats = analyze_sector_performance(processed_laps)
    
    if sector_stats:
        # Create sector comparison visualization
        create_sector_comparison_chart(sector_stats)
        
        # Add driver-specific sector analysis
        render_driver_sector_analysis(session_id, processed_laps)

def create_sector_comparison_chart(sector_stats):
    """Create an enhanced sector comparison visualization."""
    sector_data = pd.DataFrame({
        'Sector': ['Sector 1', 'Sector 2', 'Sector 3'],
        'Average Time': [
            sector_stats.get('sector1_avg', 0),
            sector_stats.get('sector2_avg', 0),
            sector_stats.get('sector3_avg', 0)
        ],
        'Best Time': [
            sector_stats.get('sector1_min', 0),
            sector_stats.get('sector2_min', 0),
            sector_stats.get('sector3_min', 0)
        ],
        'Standard Deviation': [
            sector_stats.get('sector1_std', 0),
            sector_stats.get('sector2_std', 0),
            sector_stats.get('sector3_std', 0)
        ]
    })
    
    fig = go.Figure()
    
    # Add bars for average time
    fig.add_trace(go.Bar(
        name='Average Time',
        x=sector_data['Sector'],
        y=sector_data['Average Time'],
        marker_color='#1E88E5'
    ))
    
    # Add bars for best time
    fig.add_trace(go.Bar(
        name='Best Time',
        x=sector_data['Sector'],
        y=sector_data['Best Time'],
        marker_color='#ff1801'
    ))
    
    fig.update_layout(
        title="Sector Time Comparison",
        xaxis_title="Sector",
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

def render_driver_sector_analysis(session_id, processed_laps):
    """Render driver-specific sector analysis."""
    drivers_df, error = F1DataAPI.get_driver_list(session_id)
    if error:
        st.error(f"Error loading driver data: {error}")
        return
        
    if drivers_df and len(drivers_df) > 0:
        if isinstance(drivers_df, list):
            drivers_df = pd.DataFrame(drivers_df)
        
        st.subheader("Driver Sector Performance")
        
        selected_driver = st.selectbox(
            "Select Driver",
            options=drivers_df['driver_number'].tolist(),
            format_func=lambda x: f"{drivers_df[drivers_df['driver_number'] == x]['driver_name'].iloc[0]} ({x})"
        )
        
        driver_sectors = processed_laps[processed_laps['driver_number'] == selected_driver]
        if not driver_sectors.empty:
            create_driver_sector_chart(driver_sectors, drivers_df, selected_driver)

def create_driver_sector_chart(driver_sectors, drivers_df, selected_driver):
    """Create an enhanced driver sector visualization."""
    driver_name = drivers_df[drivers_df['driver_number'] == selected_driver]['driver_name'].iloc[0]
    
    fig = go.Figure()
    
    # Add sector 1 times
    fig.add_trace(go.Scatter(
        x=driver_sectors['lap_number'],
        y=driver_sectors['sector1_time'],
        mode='lines+markers',
        name='Sector 1',
        line=dict(color='#ff1801', width=2),
        marker=dict(size=6)
    ))
    
    # Add sector 2 times
    fig.add_trace(go.Scatter(
        x=driver_sectors['lap_number'],
        y=driver_sectors['sector2_time'],
        mode='lines+markers',
        name='Sector 2',
        line=dict(color='#1E88E5', width=2),
        marker=dict(size=6)
    ))
    
    # Add sector 3 times
    fig.add_trace(go.Scatter(
        x=driver_sectors['lap_number'],
        y=driver_sectors['sector3_time'],
        mode='lines+markers',
        name='Sector 3',
        line=dict(color='#4CAF50', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title=f"Sector Times Progression - {driver_name}",
        xaxis_title="Lap Number",
        yaxis_title="Sector Time (seconds)",
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