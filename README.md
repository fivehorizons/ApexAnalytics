# F1 Data Analysis Dashboard

A comprehensive Formula 1 data visualization and analysis dashboard built with Streamlit. This application provides real-time insights into F1 race data, including lap times, driver comparisons, team performance, and predictive analytics.

## Features

### 📊 Dashboard
- Real-time session data visualization
- Key performance metrics
- Interactive charts and tables
- Session information display

### 🏁 Race Analysis
- Detailed lap time analysis
- Sector time comparison
- Performance trends
- Data export capabilities

### 🏎️ Driver Comparison
- Head-to-head driver comparison
- Lap time distribution analysis
- Performance metrics comparison
- Consistency analysis

### 👥 Team Performance
- Team rankings and statistics
- Performance comparison
- Consistency analysis
- Detailed team insights

### 🔮 Predictive Analytics
- Lap time prediction using machine learning
- Performance trend analysis
- Race simulation
- Feature importance analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/f1-dashboard.git
cd f1-dashboard
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Select a session from the sidebar to begin analysis

## Project Structure

```
f1-dashboard/
├── app.py                 # Main application file
├── pages/                 # Individual page modules
│   ├── dashboard.py
│   ├── race_analysis.py
│   ├── driver_comparison.py
│   ├── team_performance.py
│   └── predictive_analytics.py
├── utils.py              # Utility functions
├── styles/               # CSS and styling
│   └── main.css
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Dependencies

- Python 3.8+
- Streamlit 1.31.1
- Pandas 2.2.0
- NumPy 1.26.3
- Plotly 5.18.0
- scikit-learn 1.4.0
- Additional dependencies in requirements.txt

## Performance Optimizations

- Efficient data processing with pandas vectorized operations
- Optimized statistical calculations using numpy
- Proper memory management and caching
- Streamlined API interactions
- Enhanced error handling and logging

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Formula 1 for providing the data
- Streamlit for the amazing framework
- The F1 community for inspiration

## Support

For support, please open an issue in the GitHub repository or contact the maintainers. 