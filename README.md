# BrandonCFBRatings 🏈📊

A modular college football ratings engine built by Brandon with help from Copilot.  
Implements Colley, Massey, Elo, and Hybrid methods with JSON caching, CLI tools, and a Streamlit dashboard.

## Features
- **Multiple rating systems**: Colley, Massey, Elo, Hybrid
- **JSON caching**: avoid repeated API calls
- **CLI app**: quick rankings in terminal
- **Streamlit dashboard**: interactive visualization
- **Advanced insights**: Strength of Schedule, Momentum, and **PPoints**

## PPoints
A custom metric rewarding scheduling difficulty and performance

## Usage
```bash
# CLI
python -m apps.cli --year 2025 --method hybrid

# Streamlit
streamlit run apps/streamlit_app.py