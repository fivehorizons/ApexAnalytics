import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from utils import F1DataAPI
from pages.dashboard import render_dashboard
from pages.race_analysis import render_race_analysis
from pages.driver_comparison import render_driver_comparison
from pages.team_performance import render_team_performance
from pages.predictive_analytics import render_predictive_analytics
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Handle matplotlib warnings
try:
    import matplotlib
    matplotlib.use('Agg')  # Use Agg backend
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError as e:
    logger.warning(f"Matplotlib 3D support not available: {e}")
    logger.info("Some 3D visualizations may not be available")

# Set page config
st.set_page_config(
    page_title="F1 Data Analysis Dashboard",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
try:
    with open('styles/main.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except Exception as e:
    logger.error(f"Error loading CSS: {e}")
    # Provide fallback minimal styling
    st.markdown("""
    <style>
    .main { padding: 1rem; }
    h1 { color: #ff1801; }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables with error handling."""
    try:
        if 'available_sessions' not in st.session_state:
            st.session_state.available_sessions = []
        if 'selected_session' not in st.session_state:
            st.session_state.selected_session = None
        if 'error_count' not in st.session_state:
            st.session_state.error_count = 0
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = None
        if 'api_health' not in st.session_state:
            st.session_state.api_health = True
            
        # Initialize session state for widgets that might be accessed before creation
        # This helps prevent "$$WIDGET_ID" key errors
        if 'analysis_selection' not in st.session_state:
            st.session_state.analysis_selection = "Dashboard"
            
        # Pre-initialize any selectbox widgets that might cause issues
        # Initialize with default values rather than arbitrary integers
        widget_keys = {
            "driver1": 0,
            "driver2": 0, 
            "team1_selector": 0, 
            "team2_selector": 0,
            "session_selector": 0,
            "dc_driver1_idx": 0,
            "dc_driver2_idx": 0,
            "dc_initialized": False
        }
        for key, default_value in widget_keys.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
                
    except Exception as e:
        logger.error(f"Error initializing session state: {str(e)}")
        st.error("Error initializing application. Please refresh the page.")

def clear_cache():
    """Clear all cached data."""
    try:
        st.cache_data.clear()
        st.session_state.last_refresh = datetime.now()
        # Use st.rerun instead of experimental_rerun
        st.rerun()
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        st.error("Error clearing cache. Please try again.")

def check_api_health():
    """Check if the F1 API is responsive."""
    try:
        _, error = F1DataAPI.get_available_sessions()
        st.session_state.api_health = error is None
        return st.session_state.api_health
    except Exception as e:
        logger.error(f"Error checking API health: {str(e)}")
        st.session_state.api_health = False
        return False

@st.cache_data(ttl=600)
def get_available_sessions_cached():
    """Cached wrapper for getting available sessions."""
    sessions, error = F1DataAPI.get_available_sessions()
    return sessions, error

def render_sidebar():
    """Render the sidebar with session selection and controls."""
    with st.sidebar:
        st.title("F1 Data Analysis")
        
        # Add navigation
        st.subheader("Navigation")
        
        # Define the navigation options
        nav_options = [
            {"name": "Dashboard", "icon": "📊"},
            {"name": "Race Analysis", "icon": "🏁"},
            {"name": "Driver Comparison", "icon": "🏎️"},
            {"name": "Team Performance", "icon": "🏆"},
            {"name": "Predictive Analytics", "icon": "📈"}
        ]
        
        # Add navigation links with styling
        for option in nav_options:
            # Check if this is the current selection
            is_selected = st.session_state.analysis_selection == option["name"]
            
            # Apply custom styling to show which option is selected
            button_text = f"{option['icon']} {option['name']}"
            
            # Create a custom style string for the button
            if is_selected:
                # Custom styling for the button container
                st.markdown(
                    f"""
                    <style>
                    [data-testid="stButton"][aria-describedby="nav_{option['name'].lower().replace(' ', '_')}"] button {{
                        background-color: rgba(255, 24, 1, 0.1) !important;
                        color: #ff1801 !important;
                        font-weight: bold !important;
                        border-left: 4px solid #ff1801 !important;
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True
                )
            
            # Create the button with consistent styling
            if st.button(
                button_text,
                key=f"nav_{option['name'].lower().replace(' ', '_')}",
                use_container_width=True,
                help=f"View {option['name']} page"
            ):
                st.session_state.analysis_selection = option["name"]
                # Force a rerun to update the UI
                st.rerun()
        
        st.divider()
        
        # Add refresh and health check buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Data"):
                clear_cache()
        with col2:
            if st.button("🏥 Check API"):
                if check_api_health():
                    st.success("API is healthy")
                else:
                    st.error("API is not responding")
        
        # Show API health status
        if not st.session_state.api_health:
            st.error("⚠️ API Connection Issues")
            if st.button("Reset Error Counter"):
                st.session_state.error_count = 0
                st.rerun()
        
        # Fetch available sessions with retry mechanism
        retry_count = 0
        max_retries = 3
        sessions = None
        
        while retry_count < max_retries and not sessions:
            sessions, error = F1DataAPI.get_available_sessions()
            if error:
                if "timeout" in str(error).lower():
                    retry_count += 1
                    st.warning(f"Retrying to fetch sessions... ({retry_count}/{max_retries})")
                    time.sleep(1)
                else:
                    logger.error(f"Error fetching sessions: {str(error)}")
                    st.error("Failed to fetch available sessions. Please try refreshing.")
                    break
            else:
                break
        
        if sessions:
            st.session_state.available_sessions = sessions
            
            # Format session display names with error handling
            def format_session_name(session):
                try:
                    # Try different possible field names
                    name = session.get('session_name') or session.get('meeting_name') or session.get('name', 'Unknown')
                    type_name = session.get('session_type') or session.get('type', 'Unknown')
                    date = session.get('date', '').split('T')[0] if 'T' in session.get('date', '') else session.get('date', '')
                    return f"{name} - {type_name} ({date})"
                except Exception as e:
                    logger.warning(f"Error formatting session name: {e}")
                    return "Unknown Session"
            
            # Callback function to update session state when selection changes
            def on_session_select():
                selected_idx = st.session_state.session_selector
                if sessions and len(sessions) > selected_idx:
                    st.session_state.selected_session = sessions[selected_idx]
                else:
                    st.session_state.selected_session = None
            
            # Session selection with error handling
            try:
                if not sessions:
                    st.warning("No sessions available")
                    return
                
                # Ensure we have a valid index for the session selector
                if 'session_selector' in st.session_state and isinstance(st.session_state.session_selector, int):
                    default_idx = min(st.session_state.session_selector, len(sessions) - 1)
                else:
                    default_idx = 0
                
                selected_idx = st.selectbox(
                    "Select Session",
                    range(len(sessions)),
                    format_func=lambda i: format_session_name(sessions[i]),
                    help="Choose a session to analyze",
                    key="session_selector",
                    index=default_idx,
                    on_change=on_session_select
                )
                
                # Only update session state if sessions list is not empty
                if sessions and len(sessions) > selected_idx:
                    st.session_state.selected_session = sessions[selected_idx]
                    
                    # Display session details
                    with st.expander("Session Details", expanded=False):
                        # Clean the session data for display
                        display_data = {k: v for k, v in sessions[selected_idx].items() if v is not None}
                        st.json(display_data)
                else:
                    st.warning("No valid session selected")
                    st.session_state.selected_session = None
                    
            except Exception as e:
                logger.error(f"Error in session selection: {e}")
                st.error("Error displaying session selection. Please try refreshing.")
        else:
            st.warning("No sessions available. Please check your connection and try again.")

def main():
    """Main application function with enhanced error handling."""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Render sidebar
        render_sidebar()
        
        if st.session_state.selected_session:
            session_id = st.session_state.selected_session['session_id']
            
            # Import analysis modules only when needed
            from pages.dashboard import render_dashboard
            from pages.race_analysis import render_race_analysis
            from pages.driver_comparison import render_driver_comparison
            from pages.team_performance import render_team_performance
            from pages.predictive_analytics import render_predictive_analytics
            
            # Note: Radio button selection has been removed to simplify the UI
            # Navigation is now handled exclusively through the sidebar
            
            try:
                # Get the selected analysis from session state
                analysis = st.session_state.analysis_selection
                
                if analysis == "Dashboard":
                    render_dashboard(session_id, st.session_state.selected_session)
                elif analysis == "Race Analysis":
                    render_race_analysis(session_id)
                elif analysis == "Driver Comparison":
                    render_driver_comparison(session_id)
                elif analysis == "Team Performance":
                    render_team_performance(session_id)
                elif analysis == "Predictive Analytics":
                    render_predictive_analytics(session_id)
            except Exception as e:
                logger.error(f"Page rendering error: {str(e)}")
                st.error(f"Error rendering {analysis} page. Please try refreshing or selecting a different session.")
                st.exception(e)
        else:
            st.info("👈 Please select a session from the sidebar to begin analysis.")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An unexpected error occurred. Please refresh the page.")
        st.exception(e)

if __name__ == "__main__":
    main()