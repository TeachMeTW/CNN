import streamlit as st

def run_model():
    # Place your model training or inference code here
    st.write("Model is running...")
    # Example output
    result = "Hello, World!"
    st.write("Result:", result)

# Add a button to the Streamlit app
if st.button("Run Model"):
    # with st.spinner("Running..."):
    run_model()
