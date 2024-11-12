import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import pygwalker as pyg 


#sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ("Project Description", "User Guide", "Data", "PygWalker Visualization")
)

# main title and description
st.title("Anomaly Detection Dashboard")
st.markdown(
    """
    <div style='text-align: center;'>
        <p>
            This Dashboard is used to detect anomalous data. 
            The application provides an intuitive user interface for monitoring network behavior, 
            offering real-time insights and visualizations of detected anomalies. 
            Users can easily upload network data, trigger analysis, and explore results through interactive dashboards and graphs. 
            This streamlined workflow aids network administrators in identifying potential issues promptly and taking corrective actions, 
            minimizing downtime and enhancing overall network performance.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)

# Add a space between the typewriter and the image
st.markdown("---")
st.image(
    "Cisco_logo.png",
    caption="Cisco Design",
    use_column_width=True)
 

if page == "PygWalker Visualization":
    st.header('Data Visualization with Pygwalker')
    
    df = pd.read_csv("/Users/aryasastry/Downloads/CalPoly_example_files 2/iot_telemetry.csv")
    # pyg.walk(df) --> does not work because pygwalker not natively compatible with streamlit 

    # Render Pygwalker visualization as HTML
    walker_html = pyg.walk(df).to_html()
    components.html(walker_html, height=600, scrolling=True)


    # # Title for the Streamlit app
    # st.title("PyGWalker with Streamlit - Interactive Example")

    # # Display the dataset
    # st.write("### Sample Data")
    # st.dataframe(df)