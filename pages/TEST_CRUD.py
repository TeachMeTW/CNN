import streamlit as st
import json
from db_utils.db import (
    create_document, 
    read_document, 
    read_documents, 
    update_document, 
    delete_document
)

st.title("Test MongoDB CRUD Operations")

# Input for dynamic collection name
collection_name = st.text_input("Enter collection name:", value="default_collection")

# Select the CRUD operation to test
operation = st.selectbox("Select an operation", ["Create", "Read", "Update", "Delete", "List Documents"])

# Utility to safely parse JSON input
def parse_json_input(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        st.error("Invalid JSON input.")
        return None

# Perform actions based on selected operation
if operation == "Create":
    st.subheader("Create Document")
    json_data = st.text_area("Enter JSON data for new document:", '{"name": "Alice", "age": 30}')
    if st.button("Create Document"):
        data = parse_json_input(json_data)
        if data is not None:
            inserted_id = create_document(data, collection_name)
            st.success(f"Document inserted with ID: {inserted_id}")

elif operation == "Read":
    st.subheader("Read Document")
    json_query = st.text_area("Enter JSON query to find a document:", '{"name": "Alice"}')
    if st.button("Find Document"):
        query = parse_json_input(json_query)
        if query is not None:
            document = read_document(query, collection_name)
            if document:
                st.json(document)
            else:
                st.warning("No document found matching the query.")

elif operation == "Update":
    st.subheader("Update Document")
    json_query = st.text_area("Enter JSON query to match document:", '{"name": "Alice"}')
    json_update = st.text_area("Enter JSON data to update:", '{"age": 31}')
    if st.button("Update Document"):
        query = parse_json_input(json_query)
        new_data = parse_json_input(json_update)
        if query is not None and new_data is not None:
            count = update_document(query, new_data, collection_name)
            st.success(f"Number of documents updated: {count}")

elif operation == "Delete":
    st.subheader("Delete Document")
    json_query = st.text_area("Enter JSON query to match document to delete:", '{"name": "Alice"}')
    if st.button("Delete Document"):
        query = parse_json_input(json_query)
        if query is not None:
            count = delete_document(query, collection_name)
            st.success(f"Number of documents deleted: {count}")

elif operation == "List Documents":
    st.subheader("List Documents")
    json_query = st.text_area("Enter JSON query to filter documents (leave empty for all):", '{}')
    if st.button("List Documents"):
        query = parse_json_input(json_query)
        # If parsing failed, do not proceed.
        if query is not None:
            documents = read_documents(query, collection_name)
            if documents:
                st.write(f"Found {len(documents)} documents:")
                for doc in documents:
                    st.json(doc)
            else:
                st.warning("No documents found for the provided query.")
