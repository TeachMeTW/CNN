import streamlit as st
import pandas as pd
import pygwalker as pyg 
import streamlit.components.v1 as components
from pygwalker.api.streamlit import StreamlitRenderer


# # Set page configuration
# st.set_page_config(layout="wide")

# # Title of the app
# st.markdown("<h1 style='text-align: center; background-color: #333; color: white; padding: 10px;'>Data Analysis App Wireframe</h1>", unsafe_allow_html=True)

# # Main sections
# st.markdown("### Key Metrics")
# st.write("Metric Cards Area")
# st.empty()  # Placeholder for future content

# st.markdown("### Time-Series Chart")
# st.write("Time-Series Line Chart Placeholder")
# st.empty()  # Placeholder for future content

# st.markdown("### Bar Chart")
# st.write("Bar Chart Placeholder")
# st.empty()  # Placeholder for future content


st.header('Data Visualization with Pygwalker')

df = pd.read_csv('./data/food.csv')
# pyg.walk(df) --> does not work because pygwalker not natively compatible with streamlit 

# Render Pygwalker visualization as HTML
walker_html = pyg.walk(df).to_html()
components.html(walker_html, width=800, height=600, scrolling=True)

# st.set_page_config(page_title="PyGWalker in Streamlit - Data Analysis", layout="wide")
# st.title("PyGWalker in Streamlit Demo")
 
# @st.cache_resource
# def get_pyg_renderer() -> StreamlitRenderer:
#     kernel_computation=True
#     df = pd.read_csv("./data/iot_telemetry.csv")
#     st.write(df)
#     return StreamlitRenderer(df, spec="./gw_config.json", spec_io_mode="rw")
 
# renderer = get_pyg_renderer()
