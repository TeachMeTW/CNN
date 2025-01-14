import streamlit as st 

def show_documentation():
    st.title("Documentation for Anomaly Detection Streamlit Dashboard")

    # Overview of the App
    st.header("Overview of the App")

    st.subheader("Purpose")
    st.write(
        "The purpose of this app is to allow all employees at Cisco easily access and utilize the anomaly detection workflow "
        "used for Cisco’s hardware sensor products. The interactive dashboard is user-friendly and eliminates the "
        "need to have a technical background, enabling both technical or non-technical employees to discover significant "
        "findings about Cisco products."
    )

    st.subheader("Key Features and Goals")
    st.write(
        """
        - **Real-Time Metrics**: Displays defect rates, anomaly percentages, and total detected anomalies.  
        - **Data Visualization**: Interactive charts and tables to explore anomaly patterns and distributions.  
        - **Uploading Data**: Allows CSV data upload to MongoDB for storage and analysis.  
        - **IoT AutoEncoder Tuning**: Users can train and update custom AutoEncoder models with adjustable hyperparameters.  
        - **Interactive Exploration**: Pygwalker integration for in-depth data exploration. Can select certain settings.
        """
    )

    # Project Details
    st.header("Project Details")

    st.subheader("Context and Background of the Project")
    st.write(
        "At the start of the project, Cisco provided us with a functional anomaly detection workflow utilizing AI and ML, "
        "which was limited to a Python environment. Our interactive dashboard builds on this foundation by automating the anomaly detection workflows "
        "and making them accessible through a user-friendly interface. No technical background is necessary to use the dashboard and see the results of the anomaly detection workflows "
        "The dashboard simplifies complex data analysis, improving usability, enhancing hardware reliability, and increasing operational efficiency."
    )

    st.subheader("Relevant links to resources, research, or dicussions related to the project.")

    # Setup Guide 
    st.header("Setup Guide")
    
    st.subheader("Requisites")
    st.write(
        "1. Python 3.8+ "
        "2. MongoDB for data storage and management"
        "3. Dependencies specified in requirements.txt"
    )

    st.subheader("Deployment Information")
    st.write(
        "Deployment Method: The app is containerized using DOcker, ensuring consistency across various environments. "
        "Hosting Services: The app is deployed on Streamlit which allows users to locally run the apple without requiring local installations. "
            "cisconeural.net is the name of the interactive dashboard website. "
        "CI/CD Pipelines: Currently, we are setting up CI/CD pipelines. "
    )

    # Contributing Guidelines
    st.header("Contributing Guidelines")

    st.subheader("How to Contribute to the Project")
    st.write(
        "1. Fork the project on Github and clone your fork locally."
        "2. Create a new branch for your changes using git checkout. "
        "3. To add changes, use git add, git commit -m 'insert message', and git push."
        "4. Use the command in the readME file to run the site locally. "
    )

    # FAQ/Troubleshooting
    st.header("FAQ/Troubleshooting")

    st.write(
        "1. CSV Files Not Uploading Correctly"
        "   a. Ensure files have a .csv extension"
        "   b. Check for invalid data or formatting issues in the CSV"
        "   c. Verify that the file size does not exceed any limits"
        
        "2. Streamlit App Crashes or Does Not Start"
        "   a. Ensure all dependencies included in the requirements.txt are installed"
        
        "3. Docker Container Issues"
        "   a. Rebuild the Docker Container"

        "4. Incorrect or Missing Anomaly Detection Results"
        "   a. Verify the uploaded data is correct and matches expected formats"    
        "   b. Check the hyperparameters used for model training"
    )


    def main():
        show_documentation()

    if __name__ == "__main__":
        main()