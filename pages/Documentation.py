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
          <a style="color: #1a73e8;" href="#machine-learning">Machine Learning</a>
          <ul>
            <li><a style="color: #1a73e8;" href="#model-configuration-and-training">Model Configuration and Training</a></li>
            <li><a style="color: #1a73e8;" href="#data-exploration-and-visualization">Data Exploration and Visualization</a></li>
            <li><a style="color: #1a73e8;" href="#anomaly-detection">Anomaly Detection</a></li>
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
      The <strong>IoT Data Management & AutoEncoder Dashboard</strong> is a user‑friendly tool designed to help you manage your IoT sensor data and detect anomalies using machine learning.
    </p>
    <p>
      The dashboard has two main parts:
    </p>
    <ul>
      <li><strong>Data Management:</strong> Easily upload data, edit records, and view clear statistics and visualizations.</li>
      <li><strong>Machine Learning:</strong> Set up and train an AutoEncoder model to spot unusual sensor behavior.</li>
    </ul>
    <p>
      This guide explains every feature in plain language so that even if you’re not technical, you can use the system with confidence.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # ----------------------------
    # DATA MANAGEMENT SECTION
    # ----------------------------
    anchored_heading(2, "Data Management", "data-management")
    st.markdown("""
    <p>
      The Data Management section helps you handle your IoT data without any hassle. Whether you’re adding new data or updating existing records, everything is designed to be simple and straightforward.
    </p>
    """, unsafe_allow_html=True)

    # CSV Uploader
    anchored_heading(3, "CSV Uploader", "csv-uploader")
    st.markdown("""
    <p>
      <strong>Purpose:</strong> Quickly import your CSV files into the dashboard.
    </p>
    <ul>
      <li><strong>Preview Your Data:</strong> Check the first few rows of your file to make sure it looks correct.</li>
      <li><strong>Easy Import:</strong> Once you confirm, the data is automatically added to your database.</li>
    </ul>
    <p>
      <strong>Ideal For:</strong> Anyone with sensor data stored in CSV format who wants a no-fuss way to get that data into the system.
    </p>
    """, unsafe_allow_html=True)

    # Manual Data Editor
    anchored_heading(3, "Manual Data Editor", "manual-data-editor")
    st.markdown("""
    <p>
      <strong>Purpose:</strong> Edit your data directly within the application.
    </p>
    <ul>
      <li><strong>Update Records Easily:</strong> Change data, add new entries, or delete old ones with just a few clicks.</li>
      <li><strong>No Technical Jargon:</strong> Everything is done through a simple table—no complex database commands required.</li>
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
    # MACHINE LEARNING SECTION
    # ----------------------------
    anchored_heading(2, "Machine Learning", "machine-learning")
    st.markdown("""
    <p>
      The Machine Learning section guides you through setting up and using an AutoEncoder model to detect anomalies in your data.
    </p>
    """, unsafe_allow_html=True)

    # Model Configuration and Training
    anchored_heading(3, "Model Configuration and Training", "model-configuration-and-training")
    st.markdown("""
    <p>
      <strong>Purpose:</strong> Prepare and train an AutoEncoder model.
    </p>
    <ul>
      <li><strong>Select Your Devices:</strong> Choose which sensor data should be used for training.</li>
      <li><strong>Adjust Settings:</strong> Tweak options such as the number of neurons, dropout rates, and training time.</li>
      <li><strong>Train the Model:</strong> The system learns what “normal” looks like by compressing your data into a simpler form.</li>
    </ul>
    <p>
      <strong>Ideal For:</strong> Users who want a clear, step-by-step process to set up and train a machine learning model.
    </p>
    """, unsafe_allow_html=True)

    # Data Exploration and Visualization in ML
    anchored_heading(3, "Data Exploration and Visualization", "data-exploration-and-visualization")
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
    anchored_heading(3, "Anomaly Detection", "anomaly-detection")
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
      Getting started is simple:
    </p>
    <ol>
      <li><strong>Clone the Repository:</strong> Download the source code to your computer.</li>
      <li><strong>Set Up Environment Variables:</strong> Create a <code>.env</code> file and enter your settings (like database details).</li>
      <li><strong>Run the Application:</strong> Use the provided <code>run.sh</code> script or run it directly with:
        <br><code>streamlit run app.py</code>
      </li>
    </ol>
    <p>
      <strong>Note:</strong> Make sure your MongoDB server is running and that you have installed all the necessary Python libraries (listed in <code>requirements.txt</code>).
    </p>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

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
