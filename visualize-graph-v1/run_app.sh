#!/bin/bash
# Helper script to run Streamlit app with virtual environment

cd "$(dirname "$0")"
source ../venv/bin/activate
streamlit run app.py

