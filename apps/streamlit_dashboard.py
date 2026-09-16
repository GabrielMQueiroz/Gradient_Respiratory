import streamlit as st
import os

st.set_page_config(
    page_title="sttensor Workbench",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages_dir = os.path.join(os.path.dirname(__file__), "app_pages")

pages = [
    st.Page(os.path.join(pages_dir, "home.py"), title="Home", icon=":material/home:", default=True),
    st.Page(os.path.join(pages_dir, "method_comparison.py"), title="Method Comparison", icon=":material/compare:"),
    st.Page(os.path.join(pages_dir, "hypothesis_lab.py"), title="Hypothesis Lab", icon=":material/science:"),
    st.Page(os.path.join(pages_dir, "parameter_explorer.py"), title="Parameter Explorer", icon=":material/tune:"),
    st.Page(os.path.join(pages_dir, "respiratory_demo.py"), title="Respiratory Demo", icon=":material/air:"),
    st.Page(os.path.join(pages_dir, "benchmark.py"), title="Library Benchmarks", icon=":material/speed:"),
]

pg = st.navigation(pages)
pg.run()
