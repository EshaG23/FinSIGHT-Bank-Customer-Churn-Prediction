"""FinSIGHT Customer Churn Intelligence: Streamlit entry point.

Run with:  streamlit run app.py
"""
import streamlit as st

import state
import ui

st.set_page_config(page_title="FinSIGHT | Churn Intelligence", page_icon=":material/shield:",
                   layout="wide", initial_sidebar_state="expanded")

state.init()          # also sets the default theme the first time
ui.inject_css()       # variables for the CURRENT theme + shared stylesheet
service = ui.get_service()

pages = {
    "dashboard": st.Page("views/home.py", title="Overview Dashboard", icon=":material/space_dashboard:", default=True),
    "explorer": st.Page("views/explorer.py", title="Test Dataset Explorer", icon=":material/dataset:", url_path="explorer"),
    "predict": st.Page("views/predict.py", title="New Customer Prediction", icon=":material/person_search:", url_path="predict"),
    "compare": st.Page("views/compare.py", title="Model Comparison", icon=":material/compare_arrows:", url_path="compare"),
    "insights": st.Page("views/insights.py", title="Performance & Insights", icon=":material/query_stats:", url_path="insights"),
    "retention": st.Page("views/retention.py", title="Retention Insights", icon=":material/favorite:", url_path="retention"),
    "history": st.Page("views/history.py", title="Prediction History", icon=":material/history:", url_path="history"),
}
nav = st.navigation(list(pages.values()), position="hidden")

with st.sidebar:
    ui.sidebar_brand(service)
    for page in pages.values():
        st.page_link(page)
    ui.sidebar_footer(service)

_, top_right = st.columns([5, 1.25])
with top_right:
    ui.theme_button_topbar()

nav.run()
