"""
CSS styles for the F1 Data Visualization app.
This file contains all the CSS styles used in the app to keep the main app.py file cleaner.
"""

# Session card style
SESSION_CARD_STYLE = """
<style>
.session-card {
    background-color: #fff;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.metric-container {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
}
.metric-item {
    background-color: #f9f9f9;
    border-radius: 5px;
    padding: 15px;
    margin: 5px;
    flex: 1;
    min-width: 150px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    text-align: center;
}
.metric-value {
    font-size: 24px;
    font-weight: bold;
    color: #ff1801;
}
.metric-label {
    font-size: 14px;
    color: #666;
    margin-top: 5px;
}
</style>
"""

# Stats grid style
STATS_GRID_STYLE = """
<style>
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 15px;
    margin-top: 20px;
}
.stat-card {
    background-color: #f9f9f9;
    border-radius: 5px;
    padding: 15px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    text-align: center;
}
.stat-value {
    font-size: 24px;
    font-weight: bold;
    color: #ff1801;
}
.stat-label {
    font-size: 14px;
    color: #666;
    margin-top: 5px;
}
</style>
"""

# Driver comparison style
DRIVER_COMPARISON_STYLE = """
<style>
.stats-comparison {
    display: flex;
    justify-content: space-between;
    gap: 20px;
    margin-top: 20px;
}
.driver-stats {
    flex: 1;
    background-color: #f9f9f9;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}
.stats-header {
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 10px;
    text-align: center;
}
.stat-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    padding-bottom: 5px;
    border-bottom: 1px solid #eee;
}
.stat-label {
    color: #666;
}
.stat-value {
    font-weight: bold;
}
.diff-better {
    color: green;
}
.diff-worse {
    color: red;
}
</style>
"""

# Team comparison style
TEAM_COMPARISON_STYLE = """
<style>
.team-comparison {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;
    margin-bottom: 20px;
}
.team-stats {
    background-color: white;
    border-radius: 5px;
    padding: 15px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}
.team-header {
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 15px;
    text-align: center;
}
.team-stat-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 10px;
    padding-bottom: 5px;
    border-bottom: 1px solid #eee;
}
.team-stat-label {
    color: #666;
}
.team-stat-value {
    font-weight: bold;
}
.diff-better {
    color: green;
}
.diff-worse {
    color: red;
}
</style>
"""

# Loading spinner style
LOADING_SPINNER_STYLE = """
<style>
.loading-spinner {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100px;
}
.spinner {
    border: 4px solid rgba(0, 0, 0, 0.1);
    width: 36px;
    height: 36px;
    border-radius: 50%;
    border-left-color: #ff1801;
    animation: spin 1s linear infinite;
}
@keyframes spin {
    0% {
        transform: rotate(0deg);
    }
    100% {
        transform: rotate(360deg);
    }
}
</style>
""" 