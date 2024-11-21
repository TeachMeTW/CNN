import streamlit as st
import pandas as pd
import numpy as np
import pygwalker as pyg
import streamlit.components.v1 as components
from pygwalker.api.streamlit import StreamlitRenderer
import logging
from scipy import stats
import requests

# Set up logging
logging.basicConfig(
    filename="app_log.txt",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

# Log app start
logging.info("Streamlit app started.")

# Set Streamlit page configuration
st.set_page_config(
    page_title="Cisco Parts Anomaly Detection with CNN",
    layout="wide"
)

# App Title
st.markdown(
    "<h1 style='text-align: center; color: #2E8B57;'>Cisco Parts Anomaly Detection Dashboard</h1>",
    unsafe_allow_html=True,
)

# Generate dummy data
@st.cache_data
def generate_dummy_data():
    np.random.seed(42)
    num_parts = 200
    part_ids = [f"Cisco-{1000 + i}" for i in range(num_parts)]
    part_names = [f"Part-{i}" for i in range(num_parts)]
    categories = np.random.choice(
        ["Routers", "Switches", "Firewalls", "Access Points"], size=num_parts
    )
    quantity_sold = np.random.randint(50, 1000, size=num_parts)
    # Generate defect rates with some anomalies
    defect_rate = np.random.normal(loc=2.0, scale=0.5, size=num_parts)
    # Inject anomalies
    anomaly_indices = np.random.choice(num_parts, size=10, replace=False)
    defect_rate[anomaly_indices] += np.random.uniform(3, 5, size=10)  # Significant increase
    last_updated = pd.date_range(start="2023-01-01", periods=num_parts, freq="D")
    
    df = pd.DataFrame({
        "Part_ID": part_ids,
        "Part_Name": part_names,
        "Category": categories,
        "Quantity_Sold": quantity_sold,
        "Defect_Rate": defect_rate,
        "Last_Updated": last_updated
    })
    
    return df

df = generate_dummy_data()

# Log the dataframe info
logging.info(f"Dataframe created with shape: {df.shape}")

# Perform Anomaly Detection using Z-Score on Defect Rate
def detect_anomalies(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))
    anomalies = z_scores > threshold
    return anomalies

df["Anomaly"] = detect_anomalies(df["Defect_Rate"])

# Calculate key metrics
average_defect_rate = df["Defect_Rate"].mean()
total_anomalies = df["Anomaly"].sum()
anomaly_percentage = (total_anomalies / len(df)) * 100

# Display Key Metrics
st.markdown("### Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Average Defect Rate (%)", f"{average_defect_rate:.2f}")
col2.metric("Total Anomalies Detected", f"{total_anomalies}")
col3.metric("Anomaly Percentage (%)", f"{anomaly_percentage:.2f}")

# Display Defect Rate Distribution with Anomalies Highlighted
st.markdown("### Defect Rate Distribution")
defect_chart = (
    df.set_index("Part_ID")[["Defect_Rate", "Anomaly"]]
    .sort_values("Defect_Rate")
)
st.bar_chart(defect_chart["Defect_Rate"])

# Highlight Anomalies
st.markdown("#### Anomalous Parts")
anomalies_df = df[df["Anomaly"]]
st.dataframe(anomalies_df[["Part_ID", "Part_Name", "Category", "Defect_Rate", "Last_Updated"]])

# Data Visualization with Pygwalker
st.header("Interactive Data Exploration with Pygwalker")

# Initialize Pygwalker Renderer with the dataset
try:
    renderer = StreamlitRenderer(dataset=df)
    renderer.walk()
    logging.info("Pygwalker visualization rendered successfully.")
except Exception as e:
    logging.error(f"Error rendering Pygwalker visualization: {e}")
    st.error("An error occurred while rendering the Pygwalker visualization.")

# Optional: Display the raw data
with st.expander("Show Raw Data"):
    st.dataframe(df)

# Sidebar for Additional Filters (Optional)
st.sidebar.header("Filters")
selected_category = st.sidebar.multiselect(
    "Select Category:",
    options=df["Category"].unique(),
    default=df["Category"].unique()
)

if selected_category:
    filtered_df = df[df["Category"].isin(selected_category)]
else:
    filtered_df = df

# Update Key Metrics based on filters
average_defect_rate_filtered = filtered_df["Defect_Rate"].mean()
total_anomalies_filtered = filtered_df["Anomaly"].sum()
anomaly_percentage_filtered = (total_anomalies_filtered / len(filtered_df)) * 100

with st.sidebar:
    st.markdown("### Filtered Metrics")
    st.metric("Average Defect Rate (%)", f"{average_defect_rate_filtered:.2f}")
    st.metric("Total Anomalies Detected", f"{total_anomalies_filtered}")
    st.metric("Anomaly Percentage (%)", f"{anomaly_percentage_filtered:.2f}")

# Add a "Show Source Code" Button
st.sidebar.markdown("### Source Code")
if st.sidebar.button("Show Source Code"):
    st.sidebar.markdown("#### GitHub Repository: [TeachMeTW/CNN](https://github.com/TeachMeTW/CNN)")
    try:
        # Specify the file you want to display
        # For demonstration, we'll fetch the README.md file
        github_username = "TeachMeTW"
        repo_name = "CNN"
        file_path = "README.md"  # Change this to the desired file path within the repo
        
        # GitHub raw content URL
        raw_url = f"https://raw.githubusercontent.com/{github_username}/{repo_name}/main/{file_path}"
        
        response = requests.get(raw_url)
        if response.status_code == 200:
            code_content = response.text
            st.markdown(f"##### `{file_path}`:")
            st.code(code_content, language='markdown')
        else:
            st.error(f"Failed to fetch the file. Status code: {response.status_code}")
    except Exception as e:
        logging.error(f"Error fetching source code: {e}")
        st.error("An error occurred while fetching the source code.")

# Log completion
logging.info("App execution completed successfully.")
