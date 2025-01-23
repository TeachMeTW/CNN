import streamlit as st

# Configure the page
st.set_page_config(
    page_title="Cisco Anomaly Detection",
    page_icon=":satellite:",
    layout="centered",
)

# Header Section with Title and Logo
st.title("Cisco Anomaly Detection")
st.image("Cisco_logo.png", width=500)

# Introduction Section
st.subheader("Advanced Anomaly Detection System")
st.write("""
Welcome to the Cisco Anomaly Detection portal. Our project aims to transform 
how quality assurance engineers detect and monitor anomalies across Cisco’s 
product lines using an intuitive, user-friendly dashboard. 
""")

# Key Features Section
st.markdown("### Key Features")
st.markdown("""
- **User-Friendly Interface:** A graphical dashboard built with Streamlit for non-technical users.
- **Scalable Algorithms:** Optimized anomaly detection that extends across multiple Cisco products.
- **Real-Time Monitoring:** Immediate detection and visualization of anomalies.
- **Enhanced Efficiency:** Improved machine learning models for accurate and versatile analysis.
""")


# Call-to-Action
st.subheader("Get Started")
st.write("""
Discover how our advanced anomaly detection system can enhance productivity and 
improve product quality. Explore features, request a demo, or contact our team 
for more information.
""")

# Contact Button or Link (Optional)
if st.button("Contact Us"):
    st.write("Please email us at [TBD](mailto:your-email@example.com) for inquiries.")

# Footer
st.markdown("---")
st.caption("Not officially affiliated with Cisco. All rights reserved.")
