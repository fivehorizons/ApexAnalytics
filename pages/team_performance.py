import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from utils import F1DataAPI, process_lap_times, calculate_advanced_statistics, format_lap_time
from styles import TEAM_COMPARISON_STYLE
import time
import logging

logger = logging.getLogger(__name__)

@st.cache_data(ttl=300)
def get_optimized_lap_times(session_id):
    """Get lap times data with optimized caching."""
    lap_times_df, error = F1DataAPI.get_lap_times(session_id)
    return lap_times_df, error

@st.cache_data(ttl=300)
def get_optimized_driver_list(session_id):
    """Get driver list with optimized caching."""
    drivers_df, error = F1DataAPI.get_driver_list(session_id)
    return drivers_df, error

def render_team_performance(session_id):
    """Render the team performance page with enhanced visualizations."""
    st.header("Team Performance Analysis")
    
    # Fetch lap times with error handling and retry mechanism
    with st.spinner("Loading lap times..."):
        retry_count = 0
        max_retries = 3
        lap_times_df = None
        error = None
        
        while retry_count < max_retries and lap_times_df is None:
            lap_times_df, error = get_optimized_lap_times(session_id)
            if error and "timeout" in str(error).lower():
                retry_count += 1
                with st.status(f"Retrying... ({retry_count}/{max_retries})"):
                    time.sleep(1)
            else:
                break
                
        if error:
            st.error(f"Error loading lap times: {error}")
            return
            
        if lap_times_df is not None and not lap_times_df.empty:
            # Process lap times
            processed_laps = process_lap_times(lap_times_df)
            
            # Get driver list for team information with retry mechanism
            retry_count = 0
            drivers_df = None
            
            while retry_count < max_retries and drivers_df is None:
                drivers_df, error = get_optimized_driver_list(session_id)
                if error and "timeout" in str(error).lower():
                    retry_count += 1
                    with st.status(f"Retrying... ({retry_count}/{max_retries})"):
                        time.sleep(1)
                else:
                    break
                    
            if error:
                st.error(f"Error loading driver data: {error}")
                return
                
            if drivers_df and len(drivers_df) > 0:
                # Convert to DataFrame if needed
                if isinstance(drivers_df, list):
                    drivers_df = pd.DataFrame(drivers_df)
                
                # Merge lap data with driver info to get team names
                team_laps = pd.merge(
                    processed_laps,
                    drivers_df[['driver_number', 'team_name']],
                    on='driver_number',
                    how='left'
                )
                
                # Create tabs for different team analyses
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Team Overview",
                    "🏎️ Performance Comparison",
                    "📈 Detailed Analysis",
                    "📋 Team Summary"
                ])
                
                with tab1:
                    render_team_overview(team_laps)
                
                with tab2:
                    render_team_comparison(team_laps)
                
                with tab3:
                    render_detailed_team_analysis(team_laps)
                    
                with tab4:
                    render_team_summary(team_laps)
            else:
                st.warning("No driver data available for this session")
        else:
            st.warning("No lap time data available for this session")

def render_team_overview(team_laps):
    """Render team overview with enhanced visualizations."""
    try:
        st.subheader("Team Performance Overview")
        
        # Calculate team statistics with error handling
        team_stats = calculate_team_statistics(team_laps)
        
        if team_stats.empty:
            st.warning("No valid team statistics available.")
            return
        
        # Display team rankings
        st.markdown("### Team Rankings")
        
        # Create ranking metrics with delta indicators
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if len(team_stats) >= 2:
                fastest_delta = f"{(team_stats.iloc[1]['best_lap'] - team_stats.iloc[0]['best_lap']):.3f}s slower"
            else:
                fastest_delta = "No comparison available"
                
            st.metric(
                "Fastest Team",
                team_stats.iloc[0]['team_name'] if not team_stats.empty else "N/A",
                fastest_delta,
                help="Team with the fastest single lap time"
            )
        
        with col2:
            most_consistent_idx = team_stats['std_dev'].argmin()
            if len(team_stats) > 1:
                consistency_delta = f"{(team_stats['std_dev'].median() - team_stats.iloc[most_consistent_idx]['std_dev']):.3f}s better"
            else:
                consistency_delta = "No comparison available"
                
            st.metric(
                "Most Consistent Team",
                team_stats.iloc[most_consistent_idx]['team_name'] if not team_stats.empty else "N/A",
                consistency_delta,
                help="Team with the most consistent lap times"
            )
        
        with col3:
            most_laps_idx = team_stats['total_laps'].argmax()
            if len(team_stats) > 1:
                laps_delta = f"+{int(team_stats.iloc[most_laps_idx]['total_laps'] - team_stats['total_laps'].median())} laps"
            else:
                laps_delta = "No comparison available"
                
            st.metric(
                "Most Laps Completed",
                team_stats.iloc[most_laps_idx]['team_name'] if not team_stats.empty else "N/A",
                laps_delta,
                help="Team that completed the most laps"
            )
        
        # Create team performance overview chart
        create_team_overview_chart(team_stats)
        
        # Add performance distribution chart
        create_team_performance_distribution(team_laps)
    except Exception as e:
        st.error(f"Error rendering team overview: {str(e)}")
        st.info("Please try refreshing the page or selecting a different session.")

def calculate_team_statistics(team_laps):
    """Calculate comprehensive team statistics."""
    # First, remove any NaN values
    team_laps = team_laps.dropna(subset=['lap_time_seconds'])
    
    team_stats = team_laps.groupby('team_name').agg({
        'lap_time_seconds': ['min', 'mean', 'std', 'count', lambda x: np.percentile(x, 75) - np.percentile(x, 25)],
        'lap_number': 'max'
    }).reset_index()
    
    # Ensure all columns are properly named
    team_stats.columns = [
        'team_name', 'best_lap', 'avg_lap', 'std_dev', 'lap_count', 'iqr', 'total_laps'
    ]
    
    # Handle any potential NaN values in statistics
    team_stats = team_stats.fillna({
        'std_dev': team_stats['std_dev'].mean(),
        'iqr': team_stats['iqr'].mean()
    })
    
    # Calculate additional metrics
    team_stats['consistency_score'] = 100 * (1 - (team_stats['std_dev'] / team_stats['avg_lap']))
    team_stats['performance_index'] = (
        team_stats['best_lap'].rank() + 
        team_stats['std_dev'].rank() + 
        team_stats['avg_lap'].rank()
    ) / 3
    
    # Sort by best lap time
    team_stats = team_stats.sort_values('best_lap')
    
    return team_stats

def create_team_overview_chart(team_stats):
    """Create an enhanced team overview visualization."""
    try:
        fig = go.Figure()
        
        # Add best lap times
        fig.add_trace(go.Bar(
            name='Best Lap',
            x=team_stats['team_name'],
            y=team_stats['best_lap'],
            marker_color='#ff1801',
            text=team_stats['best_lap'].apply(lambda x: f"{x:.3f}s"),
            textposition='auto'
        ))
        
        # Add average lap times
        fig.add_trace(go.Bar(
            name='Average Lap',
            x=team_stats['team_name'],
            y=team_stats['avg_lap'],
            marker_color='#1E88E5',
            text=team_stats['avg_lap'].apply(lambda x: f"{x:.3f}s"),
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Team Performance Overview",
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
    except Exception as e:
        st.error(f"Error creating team overview chart: {str(e)}")
        st.info("This could be due to missing or invalid data. Please try refreshing the page or selecting a different session.")

def create_team_performance_distribution(team_laps):
    """Create a violin plot showing lap time distribution by team."""
    fig = go.Figure()
    
    for team in sorted(team_laps['team_name'].unique()):
        team_data = team_laps[team_laps['team_name'] == team]
        
        fig.add_trace(go.Violin(
            x=[team] * len(team_data),
            y=team_data['lap_time_seconds'],
            name=team,
            box_visible=True,
            meanline_visible=True
        ))
    
    fig.update_layout(
        title="Lap Time Distribution by Team",
        xaxis_title="Team",
        yaxis_title="Lap Time (seconds)",
        template="plotly_white",
        hoverlabel=dict(bgcolor="white"),
        showlegend=False,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_team_comparison(team_laps):
    """Render enhanced team comparison analysis."""
    st.subheader("Team Performance Comparison")
    
    # Validate input data
    if team_laps is None or team_laps.empty:
        st.warning("No valid lap data available for team comparison")
        return
        
    # Check if we have the required data for analysis
    if 'team_name' not in team_laps.columns or 'lap_time_seconds' not in team_laps.columns:
        st.warning("Missing required data columns for team comparison")
        return
    
    # Get unique teams with valid lap times
    valid_team_laps = team_laps.dropna(subset=['lap_time_seconds'])
    valid_teams = sorted(valid_team_laps['team_name'].unique())
    
    if len(valid_teams) < 2:
        st.warning("At least two teams with valid lap data are required for comparison")
        return
    
    # Team selection
    col1, col2 = st.columns(2)
    
    with col1:
        team1 = st.selectbox(
            "Select Team 1",
            options=valid_teams,
            index=0,
            key="team1_selector",
            help="Select the first team to compare"
        )
    
    with col2:
        default_idx = 1 if len(valid_teams) > 1 else 0
        team2 = st.selectbox(
            "Select Team 2",
            options=valid_teams,
            index=default_idx,
            key="team2_selector",
            help="Select the second team to compare"
        )
    
    # Filter data for selected teams and ensure we have valid data
    team1_laps = valid_team_laps[valid_team_laps['team_name'] == team1]
    team2_laps = valid_team_laps[valid_team_laps['team_name'] == team2]
    
    if team1_laps.empty or team2_laps.empty:
        st.warning(f"One or both teams have no valid lap data. Please select different teams.")
        return
    
    # Calculate team statistics with better error handling
    try:
        team1_stats = calculate_advanced_statistics(team1_laps)
        team2_stats = calculate_advanced_statistics(team2_laps)
        
        if not team1_stats or not team2_stats:
            st.warning("Could not calculate valid statistics for one or both teams")
            return
            
        # Create comparison metrics
        create_team_comparison_metrics(team1, team2, team1_stats, team2_stats)
        
        # Create detailed comparison visualizations
        create_team_comparison_charts(team1_laps, team2_laps, team1, team2)
        
        # Add sector comparison if available
        if 'sector1_time' in team1_laps.columns and 'sector1_time' in team2_laps.columns:
            create_sector_comparison(team1_laps, team2_laps, team1, team2)
    except Exception as e:
        logger.error(f"Error comparing teams: {str(e)}")
        st.error(f"An error occurred during team comparison: {str(e)}")

def create_team_comparison_metrics(team1, team2, team1_stats, team2_stats):
    """Create enhanced team comparison metrics."""
    # Check if stats dictionaries have the required fields
    required_fields = ['fastest_lap', 'average_lap', 'std_dev', 'total_laps']
    
    # Default values if stats are missing
    team1_stats = team1_stats if team1_stats else {}
    team2_stats = team2_stats if team2_stats else {}
    
    # Check for missing fields
    for field in required_fields:
        if field not in team1_stats:
            logger.warning(f"Missing field '{field}' in team1_stats")
            team1_stats[field] = 0
        if field not in team2_stats:
            logger.warning(f"Missing field '{field}' in team2_stats")
            team2_stats[field] = 0
    
    # Calculate differences
    diff_stats = {
        'fastest_diff': team1_stats['fastest_lap'] - team2_stats['fastest_lap'],
        'average_diff': team1_stats['average_lap'] - team2_stats['average_lap'],
        'consistency_diff': team1_stats['std_dev'] - team2_stats['std_dev'],
        'laps_diff': team1_stats['total_laps'] - team2_stats['total_laps']
    }
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Best Lap Difference",
            f"{abs(diff_stats['fastest_diff']):.3f}s",
            f"{team1 if diff_stats['fastest_diff'] < 0 else team2} faster",
            help="Difference in fastest lap times between teams"
        )
    
    with col2:
        st.metric(
            "Average Lap Difference",
            f"{abs(diff_stats['average_diff']):.3f}s",
            f"{team1 if diff_stats['average_diff'] < 0 else team2} faster",
            help="Difference in average lap times between teams"
        )
    
    with col3:
        st.metric(
            "Consistency Difference",
            f"{abs(diff_stats['consistency_diff']):.3f}s",
            f"{team1 if diff_stats['consistency_diff'] < 0 else team2} more consistent",
            help="Difference in lap time consistency between teams"
        )
        
    with col4:
        st.metric(
            "Laps Completed Difference",
            str(abs(int(diff_stats['laps_diff']))),
            f"{team1 if diff_stats['laps_diff'] > 0 else team2} more laps",
            help="Difference in number of laps completed"
        )

def create_team_comparison_charts(team1_laps, team2_laps, team1, team2):
    """Create enhanced team comparison visualizations."""
    # Create lap time progression comparison
    fig_progression = go.Figure()
    
    # Add team 1 lap times and rolling average
    fig_progression.add_trace(go.Scatter(
        x=team1_laps['lap_number'],
        y=team1_laps['lap_time_seconds'],
        mode='markers',
        name=f"{team1} Laps",
        marker=dict(color='rgba(255, 24, 1, 0.3)', size=6)
    ))
    
    rolling_avg1 = team1_laps['lap_time_seconds'].rolling(window=3, min_periods=1).mean()
    fig_progression.add_trace(go.Scatter(
        x=team1_laps['lap_number'],
        y=rolling_avg1,
        mode='lines',
        name=f"{team1} Trend",
        line=dict(color='#ff1801', width=2)
    ))
    
    # Add team 2 lap times and rolling average
    fig_progression.add_trace(go.Scatter(
        x=team2_laps['lap_number'],
        y=team2_laps['lap_time_seconds'],
        mode='markers',
        name=f"{team2} Laps",
        marker=dict(color='rgba(30, 136, 229, 0.3)', size=6)
    ))
    
    rolling_avg2 = team2_laps['lap_time_seconds'].rolling(window=3, min_periods=1).mean()
    fig_progression.add_trace(go.Scatter(
        x=team2_laps['lap_number'],
        y=rolling_avg2,
        mode='lines',
        name=f"{team2} Trend",
        line=dict(color='#1E88E5', width=2)
    ))
    
    fig_progression.update_layout(
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
    
    st.plotly_chart(fig_progression, use_container_width=True)
    
    # Create lap time distribution comparison
    fig_dist = go.Figure()
    
    # Add violin plot for team 1
    fig_dist.add_trace(go.Violin(
        y=team1_laps['lap_time_seconds'],
        name=team1,
        side='negative',
        line_color='#ff1801',
        fillcolor='rgba(255, 24, 1, 0.3)',
        meanline_visible=True,
        box_visible=True
    ))
    
    # Add violin plot for team 2
    fig_dist.add_trace(go.Violin(
        y=team2_laps['lap_time_seconds'],
        name=team2,
        side='positive',
        line_color='#1E88E5',
        fillcolor='rgba(30, 136, 229, 0.3)',
        meanline_visible=True,
        box_visible=True
    ))
    
    fig_dist.update_layout(
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
    
    st.plotly_chart(fig_dist, use_container_width=True)

def render_detailed_team_analysis(team_laps):
    """Render detailed team analysis with enhanced visualizations."""
    st.subheader("Detailed Team Analysis")
    
    # Calculate lap time evolution for each team
    team_evolution = calculate_team_evolution(team_laps)
    
    # Create evolution chart
    create_team_evolution_chart(team_evolution)
    
    # Create team consistency analysis
    create_team_consistency_analysis(team_laps)
    
    # Add performance improvement analysis
    create_performance_improvement_analysis(team_laps)

def calculate_team_evolution(team_laps):
    """Calculate team performance evolution over laps."""
    team_evolution = team_laps.groupby(['team_name', 'lap_number'])['lap_time_seconds'].agg([
        'mean', 'min', 'max'
    ]).reset_index()
    
    # Calculate rolling averages
    evolution_data = []
    for team in team_evolution['team_name'].unique():
        team_data = team_evolution[team_evolution['team_name'] == team].copy()
        team_data['rolling_avg'] = team_data['mean'].rolling(window=3, min_periods=1).mean()
        evolution_data.append(team_data)
    
    return pd.concat(evolution_data)

def create_team_evolution_chart(team_evolution):
    """Create enhanced team evolution visualization."""
    fig = go.Figure()
    
    for team in sorted(team_evolution['team_name'].unique()):
        team_data = team_evolution[team_evolution['team_name'] == team]
        
        # Add range of lap times
        fig.add_trace(go.Scatter(
            x=team_data['lap_number'],
            y=team_data['min'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            name=f"{team} Min"
        ))
        
        fig.add_trace(go.Scatter(
            x=team_data['lap_number'],
            y=team_data['max'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=f'rgba(128, 128, 128, 0.2)',
            showlegend=False,
            name=f"{team} Max"
        ))
        
        # Add trend line
        fig.add_trace(go.Scatter(
            x=team_data['lap_number'],
            y=team_data['rolling_avg'],
            mode='lines',
            name=team,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title="Team Performance Evolution",
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

def create_team_consistency_analysis(team_laps):
    """Create enhanced team consistency analysis."""
    # Calculate consistency metrics
    consistency_stats = team_laps.groupby('team_name').agg({
        'lap_time_seconds': ['std', 'mean', lambda x: np.percentile(x, 75) - np.percentile(x, 25)]
    }).reset_index()
    
    consistency_stats.columns = ['team_name', 'std_dev', 'mean_time', 'iqr']
    
    # Sort by standard deviation
    consistency_stats = consistency_stats.sort_values('std_dev')
    
    # Create consistency visualization
    fig = go.Figure()
    
    # Add standard deviation bars
    fig.add_trace(go.Bar(
        name='Standard Deviation',
        x=consistency_stats['team_name'],
        y=consistency_stats['std_dev'],
        marker_color='#ff1801',
        text=consistency_stats['std_dev'].apply(lambda x: f"{x:.3f}s"),
        textposition='auto'
    ))
    
    # Add IQR bars
    fig.add_trace(go.Bar(
        name='Interquartile Range',
        x=consistency_stats['team_name'],
        y=consistency_stats['iqr'],
        marker_color='#1E88E5',
        text=consistency_stats['iqr'].apply(lambda x: f"{x:.3f}s"),
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Team Consistency Analysis",
        xaxis_title="Team",
        yaxis_title="Time Variation (seconds)",
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

def create_performance_improvement_analysis(team_laps):
    """Create analysis of team performance improvement."""
    st.subheader("Performance Improvement Analysis")
    
    # Calculate improvement metrics for each team
    improvement_stats = []
    
    for team in sorted(team_laps['team_name'].unique()):
        team_data = team_laps[team_laps['team_name'] == team].copy()
        
        # Split data into first and second half
        mid_point = len(team_data) // 2
        first_half = team_data.iloc[:mid_point]
        second_half = team_data.iloc[mid_point:]
        
        # Calculate metrics
        stats = {
            'team_name': team,
            'first_half_avg': first_half['lap_time_seconds'].mean(),
            'second_half_avg': second_half['lap_time_seconds'].mean(),
            'improvement': first_half['lap_time_seconds'].mean() - second_half['lap_time_seconds'].mean()
        }
        
        improvement_stats.append(stats)
    
    improvement_df = pd.DataFrame(improvement_stats)
    
    # Create improvement visualization
    fig = go.Figure()
    
    # Add improvement bars
    fig.add_trace(go.Bar(
        x=improvement_df['team_name'],
        y=improvement_df['improvement'],
        marker_color=improvement_df['improvement'].apply(
            lambda x: '#00E676' if x > 0 else '#ff1801'
        ),
        text=improvement_df['improvement'].apply(lambda x: f"{x:.3f}s"),
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Performance Improvement (First Half vs Second Half)",
        xaxis_title="Team",
        yaxis_title="Time Improvement (seconds)",
        template="plotly_white",
        hoverlabel=dict(bgcolor="white"),
        showlegend=False,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add improvement metrics table
    st.markdown("### Detailed Improvement Metrics")
    
    improvement_df['First Half Avg'] = improvement_df['first_half_avg'].apply(format_lap_time)
    improvement_df['Second Half Avg'] = improvement_df['second_half_avg'].apply(format_lap_time)
    improvement_df['Improvement'] = improvement_df['improvement'].apply(lambda x: f"{x:.3f}s")
    
    st.dataframe(
        improvement_df[['team_name', 'First Half Avg', 'Second Half Avg', 'Improvement']],
        hide_index=True,
        column_config={
            "team_name": st.column_config.TextColumn(
                "Team",
                help="Team name",
                width="medium"
            ),
            "First Half Avg": st.column_config.TextColumn(
                "First Half Average",
                help="Average lap time in first half of session",
                width="medium"
            ),
            "Second Half Avg": st.column_config.TextColumn(
                "Second Half Average",
                help="Average lap time in second half of session",
                width="medium"
            ),
            "Improvement": st.column_config.TextColumn(
                "Improvement",
                help="Time improvement from first to second half",
                width="medium"
            )
        }
    )

def render_team_summary(team_laps):
    """Render a comprehensive team summary."""
    st.subheader("Team Performance Summary")
    
    # Calculate summary statistics
    summary_stats = calculate_team_summary(team_laps)
    
    # Display summary table
    st.dataframe(
        summary_stats,
        hide_index=True,
        column_config={
            "Team": st.column_config.TextColumn(
                "Team",
                help="Team name",
                width="medium"
            ),
            "Best Lap": st.column_config.TextColumn(
                "Best Lap",
                help="Fastest lap time",
                width="medium"
            ),
            "Average Lap": st.column_config.TextColumn(
                "Average Lap",
                help="Average lap time",
                width="medium"
            ),
            "Consistency": st.column_config.TextColumn(
                "Consistency",
                help="Lap time standard deviation",
                width="medium"
            ),
            "Total Laps": st.column_config.NumberColumn(
                "Total Laps",
                help="Number of laps completed",
                width="small"
            ),
            "Performance Score": st.column_config.ProgressColumn(
                "Performance Score",
                help="Overall performance score (0-100)",
                min_value=0,
                max_value=100,
                width="medium"
            )
        }
    )
    
    # Add download button for summary
    csv = summary_stats.to_csv(index=False)
    st.download_button(
        label="Download Summary as CSV",
        data=csv,
        file_name="team_performance_summary.csv",
        mime="text/csv"
    )

def calculate_team_summary(team_laps):
    """Calculate comprehensive team summary statistics."""
    summary = []
    
    for team in sorted(team_laps['team_name'].unique()):
        team_data = team_laps[team_laps['team_name'] == team]
        
        # Calculate basic stats
        best_lap = team_data['lap_time_seconds'].min()
        avg_lap = team_data['lap_time_seconds'].mean()
        std_dev = team_data['lap_time_seconds'].std()
        total_laps = len(team_data)
        
        # Calculate performance score (0-100)
        # Lower values are better for all metrics
        best_lap_score = 100 * (1 - (best_lap - team_laps['lap_time_seconds'].min()) / 
                               (team_laps['lap_time_seconds'].max() - team_laps['lap_time_seconds'].min()))
        consistency_score = 100 * (1 - (std_dev / avg_lap))
        laps_score = 100 * (total_laps / team_laps.groupby('team_name').size().max())
        
        # Overall performance score
        performance_score = (best_lap_score * 0.4 + consistency_score * 0.4 + laps_score * 0.2)
        
        summary.append({
            'Team': team,
            'Best Lap': format_lap_time(best_lap),
            'Average Lap': format_lap_time(avg_lap),
            'Consistency': f"{std_dev:.3f}s",
            'Total Laps': total_laps,
            'Performance Score': round(performance_score, 1)
        })
    
    return pd.DataFrame(summary).sort_values('Performance Score', ascending=False)

def create_sector_comparison(team1_laps, team2_laps, team1_name, team2_name):
    """Create sector time comparison analysis between two teams."""
    st.subheader("Sector Performance Comparison")
    
    # Validate that we have sector data
    required_sectors = ['sector1_time', 'sector2_time', 'sector3_time']
    
    for column in required_sectors:
        if column not in team1_laps.columns or column not in team2_laps.columns:
            st.warning(f"Missing {column} data for sector comparison")
            return
    
    # Prepare sector data for both teams
    try:
        # Calculate average sector times for team 1
        team1_sectors = {
            'Sector 1': team1_laps['sector1_time'].mean(),
            'Sector 2': team1_laps['sector2_time'].mean(),
            'Sector 3': team1_laps['sector3_time'].mean()
        }
        
        # Calculate average sector times for team 2
        team2_sectors = {
            'Sector 1': team2_laps['sector1_time'].mean(),
            'Sector 2': team2_laps['sector2_time'].mean(),
            'Sector 3': team2_laps['sector3_time'].mean()
        }
        
        # Create radar chart for sector comparison
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=[team1_sectors[s] for s in team1_sectors.keys()],
            theta=list(team1_sectors.keys()),
            fill='toself',
            name=team1_name,
            line_color='#ff1801'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=[team2_sectors[s] for s in team2_sectors.keys()],
            theta=list(team2_sectors.keys()),
            fill='toself',
            name=team2_name,
            line_color='#1E88E5'
        ))
        
        fig.update_layout(
            title="Average Sector Time Comparison",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    # Lower values are better for lap times
                    range=[0, max(max(team1_sectors.values()), max(team2_sectors.values())) * 1.1]
                )
            ),
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
        
        # Add sector delta metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sector1_diff = team1_sectors['Sector 1'] - team2_sectors['Sector 1']
            st.metric(
                "Sector 1 Delta",
                f"{abs(sector1_diff):.3f}s",
                f"{team1_name if sector1_diff < 0 else team2_name} faster"
            )
        
        with col2:
            sector2_diff = team1_sectors['Sector 2'] - team2_sectors['Sector 2']
            st.metric(
                "Sector 2 Delta",
                f"{abs(sector2_diff):.3f}s",
                f"{team1_name if sector2_diff < 0 else team2_name} faster"
            )
        
        with col3:
            sector3_diff = team1_sectors['Sector 3'] - team2_sectors['Sector 3']
            st.metric(
                "Sector 3 Delta",
                f"{abs(sector3_diff):.3f}s",
                f"{team1_name if sector3_diff < 0 else team2_name} faster"
            )
            
    except Exception as e:
        logger.error(f"Error in sector comparison: {str(e)}")
        st.error(f"An error occurred during sector comparison: {str(e)}") 