import streamlit as st


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
 

