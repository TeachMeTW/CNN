import streamlit as st

def show_documentation():
    # Sidebar navigation
    st.sidebar.title("Table of Contents")
    sections = [
        "Home Page",
        "Overview of the Project",
        "Project Details",
        "Setup Guide",
        "Contributing Guidelines",
        "FAQ/Troubleshooting"
    ]
    selected_section = st.sidebar.radio("Go to:", sections)

    # Display content based on the selected section
    if selected_section == "Home Page":
        st.title("What is the purpose of this documentation page?")
        st.write(
            "This documentation page contains all of the information related to our interactive dashboard "
            "the broader project. This page serves as a single source of truth and provides a complete"
            "overview for developers, contributers, and users. All answers to questions you have about the app"
            "can be found in this page. Use the Table of Contents sidebar to explore the information."
        )


    if selected_section == "Overview of the Project":
        st.title("Overview of the Project")

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

    elif selected_section == "Project Details":
        st.title("Project Details")

        st.subheader("Context and Background of the Project")
        st.write(
            "At the start of the project, Cisco provided us with a functional anomaly detection workflow utilizing AI and ML, "
            "which was limited to a Python environment. Our interactive dashboard builds on this foundation by automating the anomaly detection workflows "
            "and making them accessible through a user-friendly interface. No technical background is necessary to use the dashboard and see the results of the anomaly detection workflows. "
            "The dashboard simplifies complex data analysis, improving usability, enhancing hardware reliability, and increasing operational efficiency."
        )

        st.subheader("Relevant links to resources, research, or discussions related to the project.")
        # Add any links or resources here.

    elif selected_section == "Setup Guide":
        st.title("Setup Guide")

        st.subheader("Requisites")
        st.write(
            """
            1. Python 3.8+  
            2. MongoDB for data storage and management  
            3. Dependencies specified in requirements.txt
            """
        )

        st.subheader("Deployment Information")
        st.write(
            """
            - **Deployment Method**: The app is containerized using Docker, ensuring consistency across various environments.  
            - **Hosting Services**: The app is deployed on Streamlit, allowing users to run the app locally without requiring local installations.  
            - **CI/CD Pipelines**: Currently, we are setting up CI/CD pipelines.
            """
        )

    elif selected_section == "Contributing Guidelines":
        st.title("Contributing Guidelines")

        st.subheader("How to Contribute to the Project")
        st.write(
            """
            1. Fork the project on GitHub and clone your fork locally.  
            2. Create a new branch for your changes using `git checkout`.  
            3. To add changes, use `git add`, `git commit -m 'insert message'`, and `git push`.  
            4. Use the command in the README file to run the site locally.
            """
        )

    elif selected_section == "FAQ/Troubleshooting":
        st.title("FAQ/Troubleshooting")

        st.write(
            """
            1. **CSV Files Not Uploading Correctly**  
               - Ensure files have a `.csv` extension.  
               - Check for invalid data or formatting issues in the CSV.  
               - Verify that the file size does not exceed any limits.  

            2. **Streamlit App Crashes or Does Not Start**  
               - Ensure all dependencies included in the `requirements.txt` are installed.  

            3. **Docker Container Issues**  
               - Rebuild the Docker Container.  

            4. **Incorrect or Missing Anomaly Detection Results**  
               - Verify the uploaded data is correct and matches expected formats.  
               - Check the hyperparameters used for model training.
            """
        )

def main():
    show_documentation()

if __name__ == "__main__":
    main()
