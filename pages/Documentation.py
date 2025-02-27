import streamlit as st

def anchored_heading(level, text, anchor_id):
    """
    Outputs a heading with an invisible anchor (so that links can jump here).
    level: The heading level (e.g., 2 for h2, 3 for h3).
    text: The text of the heading.
    anchor_id: The id used for the anchor link.
    """
    st.markdown(f'<a id="{anchor_id}"></a>', unsafe_allow_html=True)
    st.markdown(f'<h{level} style="color: #333;">{text}</h{level}>', unsafe_allow_html=True)

def main():
    # Set up the page title and layout.
    st.set_page_config(page_title="IoT Data Management & AutoEncoder Dashboard Documentation", layout="wide")
    st.title("IoT Data Management & AutoEncoder Dashboard Documentation")

    # Create a Table of Contents container with a pleasant background and dark text.
    toc_html = """

      <h3 style="margin-top: 0;">Table of Contents</h3>
      <ul>
        <li><a style="color: #1a73e8;" href="#overview">Overview</a></li>
        <li>
          <a style="color: #1a73e8;" href="#data-management">Data Management</a>
          <ul>
            <li><a style="color: #1a73e8;" href="#csv-uploader">CSV Uploader</a></li>
            <li><a style="color: #1a73e8;" href="#manual-data-editor">Manual Data Editor</a></li>
            <li><a style="color: #1a73e8;" href="#advanced-collection-management">Advanced Collection Management</a></li>
            <li><a style="color: #1a73e8;" href="#data-explorer">Data Explorer</a></li>
          </ul>
        </li>
        <li>
          <a style="color: #1a73e8;" href="#model-selection">Model Selection</a>
          <ul>
            <li><a style="color: #1a73e8;" href="#model-configuration-and-training">Model Configuration and Training</a></li>
            <li><a style="color: #1a73e8;" href="#data-exploration">Data Exploration</a></li>
            <li><a style="color: #1a73e8;" href="#result-and-visualization">Results and Visualization</a></li>
            <li><a style="color: #1a73e8;" href="#anomaly-detection">Anomalies</a></li>
          </ul>
        </li>
        <li><a style="color: #1a73e8;" href="#installation-and-setup">Installation and Setup</a></li>
        <li><a style="color: #1a73e8;" href="#future-enhancements">Future Enhancements</a></li>
        <li><a style="color: #1a73e8;" href="#frequently-asked-questions">Frequently Asked Questions</a></li>
        <li><a style="color: #1a73e8;" href="#conclusion">Conclusion</a></li>
      </ul>
    </div>
    """
    st.markdown(toc_html, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # ----------------------------
    # OVERVIEW SECTION
    # ----------------------------
    anchored_heading(2, "Overview", "overview")
    st.markdown("""
    <p>
      This <strong>IoT Data Management & AutoEncoder Dashboard</strong> is a user‑friendly tool designed to help you manage and detect anomalies in your IoT sensor data using machine learning.
    </p>
    <p>
      The dashboard has two main parts:
    </p>
    <ul>
      <li><strong>Data Management:</strong> Easily upload data, modify existing entries, and view clear statistics and visualizations.</li>
      <li><strong>Machine Learning:</strong> Set up and train an AutoEncoder model to spot unusual sensor behavior.</li>
    </ul>
    <p>
      This guide explains every feature in plain language, so even if you’re not technical, you can use the system with confidence.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # ----------------------------
    # DATA MANAGEMENT SECTION
    # ----------------------------
    anchored_heading(2, "Data Management", "data-management")
    st.markdown(""" 
    <p>
      The Data Management tab in the navigation bar on the left helps you handle your IoT data without any hassle. Whether you’re adding new data or updating existing data, everything is designed to be simple and straightforward.
    </p>
    """, unsafe_allow_html=True)

    # CSV Uploader
    anchored_heading(3, "CSV Uploader", "csv-uploader")
    st.markdown("""
    <p>
      <strong>Purpose:</strong> Quickly import your CSV files into the dashboard.
    </p>

    <p>
      <strong>How to Upload a CSV:</strong>
    </p>
    <ol>
      <li><strong>Navigate to the CSV Upload Button:</strong>
        <ul>
          <li>Use the <strong>navigation bar on the left</strong> and click on <strong>Data Management</strong>.</li>
          <li>Within the <strong>Upload & Edit Data</strong> tab, find the <strong>CSV Uploader</strong> subtitle.</li>
        </ul>
      </li>
      <li><strong>Upload Your File:</strong>
        <ul>
          <li>Click the <strong>Upload CSV</strong> button.</li>
          <li>Select your CSV file from your device.</li>
          <li>After uploading, the dashboard will display a preview of the first few rows.</li>
        </ul>
      </li>
      <li><strong>Confirm & Import:</strong>
        <ul>
          <li>Review the previewed data.</li>
          <li>If everything looks correct, click <strong>Confirm Upload</strong> to add the data to your database.</li>
        </ul>
      </li>
    </ol>

    <p>
      <strong>Features:</strong>
    </p>
    <ul>
      <li><strong>Preview Before Upload:</strong> Ensure your CSV is formatted correctly before committing changes.</li>
      <li><strong>Easy Import:</strong> Once you confirm, the data is automatically added to your database.</li>
    </ul>

    <p>
      <strong>Ideal For:</strong> Anyone with sensor data stored in CSV format who wants a simple way to import data into the system.
    </p>
    """, unsafe_allow_html=True)

    # Manual Data Editor
    anchored_heading(3, "Manual Data Editor", "manual-data-editor")
    st.markdown("""
    <p>
      <strong>Purpose:</strong> Edit your data directly within the application.
    </p>
    <p>
      <strong>Locate:</strong> The Manual Document Editor is right below the CVS Uploader, which is located within the Data Management tab on the navigation bar on the left. 
    </p>
    <ul>
      <li><strong>Edit Existing Entries:</strong> Simply click on any data cell to modify its value immediately.</li>
      <li><strong>Add New Entries:</strong> To add data, click on an empty cell in the table and type in your new value.</li>
      <li><strong>Delete Entries:</strong> Each row has a delete button on the right side. Click it to remove the entry, then click "Save Changes" to confirm the deletion.</li>
      <li><strong>No Technical Jargon:</strong> Everything is managed through an intuitive table interface—no complex commands required.</li>
    </ul>
    <p>
      <strong>Ideal For:</strong> Users who need to fix or update data without writing any code.
    </p>
    """, unsafe_allow_html=True)



    # Advanced Collection Management
    anchored_heading(3, "Advanced Collection Management", "advanced-collection-management")
    st.markdown("""
    <p>
      <strong>Purpose:</strong> Keep your data organized.
    </p>
    <ul>
      <li><strong>Delete Collections:</strong> Remove entire groups of data when they’re no longer needed.</li>
      <li><strong>Merge Collections:</strong> Combine data from different sources into one organized collection.</li>
      <li><strong>Rename Collections:</strong> Change the names of your collections to keep everything clear and tidy.</li>
    </ul>
    <p>
      <strong>Ideal For:</strong> Administrators or power users who manage large sets of data.
    </p>
    """, unsafe_allow_html=True)

    # Data Explorer
    anchored_heading(3, "Data Explorer", "data-explorer")
    st.markdown("""
    <p>
      <strong>Purpose:</strong> Visualize and understand your sensor data.
    </p>
    <p>
      <strong>Locate:</strong> At the top of the Data Management page, you will see two tabs. Click on the tab labeled 'Data Explorer.' 
    </p>
    <ul>
      <li><strong>Latest Readings:</strong> Quickly check the most recent data for each sensor.</li>
      <li><strong>Detailed Analysis:</strong> Look at statistics (like minimum, maximum, and average values) for any sensor parameter.</li>
      <li><strong>Interactive Charts:</strong> See trends over time with dynamic time-series graphs.</li>
      <li><strong>Browse Data:</strong> View your data page by page for in-depth review.</li>
    </ul>
    <p>
      <strong>Ideal For:</strong> Anyone who wants to analyze their data visually and ensure it is accurate.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # ----------------------------
    # MODEL SELECTION SECTION
    # ----------------------------
    anchored_heading(2, "Model Selection", "machine-learning")
    st.markdown("""
    <p>
      The Machine Learning section guides you through setting up and using an AutoEncoder model to detect anomalies in your data.
    </p>
    <p>
      <strong>Locate: </strong> In the navigation bar on the left, click the Model Selection tab.
    </p>
    <p>
      <strong>Information: </strong>There are 4 tabs at the top of the Model Selection Page. Keep reading to learn how to use each tab. 
    </p>
    """, unsafe_allow_html=True)

    # Model Configuration and Training 
    anchored_heading(3, "1st Tab: Configuration", "model-configuration-and-training")
    st.markdown("""
    <p>
      <strong>Purpose:</strong> Prepare and train an AutoEncoder model.
    </p>
    
    <ul>
      <li><strong>Select Your Devices:</strong> Choose which sensor data should be used for training.</li>
      <li><strong>Adjust Settings:</strong> Tweak options such as the number of neurons, dropout rates, and training time.</li>
      <li><strong>Train the Model:</strong> The system learns what “normal” looks like by compressing your data into a simpler form. Once you've adjusted the settings to your liking, click the <strong>Train New Model </strong> button to start the training process.</li>
    </ul>
    <p>
      <strong>Ideal For:</strong> Users who want a clear, step-by-step process to set up and train a machine learning model.
    </p>
    """, unsafe_allow_html=True)

    # Data Exploration in ML 
    anchored_heading(3, "2nd Tab: Data Exploration", "data-exploration-and-visualization")
    st.markdown("""
    <p>
      <strong>Purpose:</strong> Ensure the data you loaded is as expected.
    </p>
    <ul>
      <li><strong>Review </strong>In this tab, you can preview your dataset and make sure it looks like it is supposed to.</li>
    </ul>
    <p>
      <strong>Ideal For:</strong> Users interested in ensuring the model is working correctly and understanding its behavior.
    </p>
    """, unsafe_allow_html=True)

    # Results and Visualization in ML
    anchored_heading(3, "3rd Tab: Results & Visualization", "data-exploration-and-visualization")
    st.markdown("""
    <p>
      <strong>Purpose:</strong> See how well the model is performing.
    </p>
    <ul>
      <li><strong>Training History:</strong> View graphs of the model’s loss and accuracy over time.</li>
      <li><strong>Latent Space:</strong> Understand how the model compresses your data.</li>
      <li><strong>Reconstruction Errors:</strong> Check how accurately the model can rebuild your data and spot errors.</li>
    </ul>
    <p>
      <strong>Ideal For:</strong> Users interested in ensuring the model is working correctly and understanding its behavior.
    </p>
    """, unsafe_allow_html=True)

    # Anomaly Detection
    anchored_heading(3, "4th Tab: Anomalies", "anomaly-detection")
    st.markdown("""
    <p>
      <strong>Purpose:</strong> Identify data points that do not follow the normal pattern.
    </p>
    <ul>
      <li><strong>Set Error Thresholds:</strong> The system automatically determines what counts as an anomaly.</li>
      <li><strong>Anomaly Summary:</strong> See a clear list and charts of all the anomalies detected.</li>
      <li><strong>Error Trends:</strong> Monitor how anomalies change over time for each sensor.</li>
    </ul>
    <p>
      <strong>Ideal For:</strong> Anyone who wants to catch unusual sensor behavior early.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

   # ----------------------------
    # INSTALLATION AND SETUP SECTION
    # ----------------------------
    anchored_heading(2, "Installation and Setup", "installation-and-setup")

    st.markdown("""
    <p>
      To use this dashboard locally, follow the steps below. Getting started is simple:
    </p>

    <ol>
      <li>
        <strong>Clone the Repository:</strong> Download <a href="https://github.com/TeachMeTW/CNN" style="color: #1a73e8;">this</a> source code to your computer. 
        <ul>
          <li>For directions on how to clone a GitHub repository, see the <a href="https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository" style="color: #1a73e8;">GitHub Cloning Guide</a>.</li>
        </ul>
      </li>

      <li>
        <strong>Set Up Environment Variables (if running locally):</strong>  
        If you are using the hosted version at <a href="https://cisconeural.net" style="color: #1a73e8;">cisconeural.net</a>, you can skip this step.
        Otherwise, create a <code>.env</code> file in the project root directory and add your settings.
        <br><br>
        For details on setting up MongoDB, see the 
        <a href="https://www.mongodb.com/docs/manual/installation/" style="color: #1a73e8;" target="_blank">MongoDB Installation Guide</a>.
      // deleted closing li
    // deleted closing ol 

    <pre><code># Streamlit Configuration
    STREAMLIT_PORT=8501
    STREAMLIT_ADDRESS=0.0.0.0

    # MongoDB Configuration
    MONGO_VERSION=6.0
    MONGO_CONTAINER_NAME=mongo
    MONGO_PORT=27017
    MONGO_ROOT_USERNAME=&lt;user&gt;
    MONGO_ROOT_PASSWORD=&lt;pass&gt;
    MONGO_INITDB_DATABASE=&lt;db&gt;
    MONGO_DATA_VOLUME=mongo_data

    # Application Configuration
    PYTHONUNBUFFERED=1
    PYTHONDONTWRITEBYTECODE=1
    MONGO_URI=mongodb://&lt;user&gt;:&lt;pass&gt;@mongo:27017/&lt;databaseurl&gt;?authSource=admin
    </code></pre>
    </li>
    </ol>

    <ol start="3">
      <li>
        <strong>Install Dependencies:</strong>  
        Run the following command to install required Python libraries:
        <pre><code>pip install -r requirements.txt</code></pre>
      </li>

      <li>
        <strong>Run the Application:</strong>  
        Use the provided <code>run.sh</code> script:
        <pre><code>./run.sh</code></pre>
        Or manually start the app with:
        <pre><code>streamlit run app.py</code></pre>
      </li>
    </ol>

    <p>
      <strong>Note:</strong> Ensure your MongoDB server is running before launching the application.
    </p>
    """, unsafe_allow_html=True)



    # ----------------------------
    # FUTURE ENHANCEMENTS SECTION
    # ----------------------------
    anchored_heading(2, "Future Enhancements", "future-enhancements")
    st.markdown("""
    <p>
      Our team is always looking to improve the application. Planned upgrades include:
    </p>
    <ul>
      <li><strong>Standalone Executable:</strong> A one-click version for easier use on any computer.</li>
      <li><strong>More ML Models:</strong> Additional machine learning options to give you deeper insights.</li>
      <li><strong>Interface Upgrades:</strong> Further polish to make everything even more user-friendly.</li>
      <li><strong>Enhanced Documentation:</strong> More tutorials and in‑app help to guide you step-by-step.</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # ----------------------------
    # FREQUENTLY ASKED QUESTIONS SECTION
    # ----------------------------
    anchored_heading(2, "Frequently Asked Questions", "frequently-asked-questions")
    st.markdown("""
    <p>
      <strong>Q1: Who should use this application?</strong><br>
      <em>A:</em> Anyone who manages IoT sensor data—whether you’re a hobbyist, run a small business, or work for a large organization.
    </p>
    <p>
      <strong>Q2: Do I need to be technical to use it?</strong><br>
      <em>A:</em> Not at all. The app is designed with clear, simple instructions so that even non‑technical users can easily navigate it.
    </p>
    <p>
      <strong>Q3: How do I upload my data?</strong><br>
      <em>A:</em> Go to the Data Management section, select the CSV Uploader, choose your CSV file, preview it, and then click “Upload.”
    </p>
    <p>
      <strong>Q4: What is an AutoEncoder?</strong><br>
      <em>A:</em> It’s a type of neural network that learns a simplified version of your data. We use it here to detect anomalies by comparing the original data with what the model reconstructs.
    </p>
    <p>
      <strong>Q5: How can I get help if I run into issues?</strong><br>
      <em>A:</em> Check the in‑app help or contact our support team using the information provided in the repository.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # ----------------------------
    # CONCLUSION SECTION
    # ----------------------------
    anchored_heading(2, "Conclusion", "conclusion")
    st.markdown("""
    <p>
      The <strong>IoT Data Management & AutoEncoder Dashboard</strong> makes it easy to manage your sensor data and detect anomalies using state‑of‑the‑art machine learning—all explained in simple, everyday language.
    </p>
    <p>
      We hope this documentation helps you get started quickly and confidently. If you have any questions or need more help, please consult the in‑app help or contact our support team.
    </p>
    <p><strong>Happy Exploring and Data Managing!</strong></p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
