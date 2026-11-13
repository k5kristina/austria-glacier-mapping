"""Entry point of the website: ``streamlit run app/streamlit_app.py`` (after ``glacier-pipeline run``)."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Austria's glaciers from space",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

pages = [
    st.Page("views/home.py", title="Introduction", icon="🏔️", default=True),
    st.Page("views/explore.py", title="Explore", icon="🛰️"),
]
st.navigation(pages, position="top").run()
