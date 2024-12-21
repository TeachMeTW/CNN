import streamlit as st
import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv

def csv_uploader(mongo_uri, db_name):
    st.header("CSV Uploader")
    
    # Collection name input
    collection_name = st.text_input("Collection Name to insert CSV data into:", value="sensor_data")

    # Upload CSV
    csv_file = st.file_uploader("Upload CSV", type="csv")
    if csv_file is not None:
        # Show CSV preview
        df = pd.read_csv(csv_file)
        st.subheader("CSV Preview")
        st.dataframe(df.head())

        if st.button("Upload to MongoDB"):
            try:
                client = MongoClient(mongo_uri)
                db = client[db_name]

                # Convert DataFrame to list of dicts
                records = df.to_dict(orient="records")

                # Insert into the chosen collection
                result = db[collection_name].insert_many(records)
                st.success(f"Inserted {len(result.inserted_ids)} documents into '{collection_name}'.")
            except Exception as e:
                st.error(f"Error uploading to MongoDB: {e}")
            finally:
                client.close()

def data_explorer(mongo_uri, db_name):
    st.header("Data Explorer")
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]

        # List all collections
        collections = db.list_collection_names()
        if not collections:
            st.info("No collections found in this database.")
            return
        
        collection_choice = st.selectbox("Select a collection", options=collections)
        if collection_choice:
            # Limit to 100 docs for performance
            cursor = db[collection_choice].find().limit(100)
            docs = list(cursor)
            st.write(f"Showing first 100 documents from **{collection_choice}** (if that many exist).")
            st.write(f"Total documents in collection may be more. Use your own queries for deeper analysis.")

            if docs:
                # Convert _id to string so we can display it easily
                for doc in docs:
                    doc["_id"] = str(doc["_id"])
                
                df = pd.DataFrame(docs)
                st.dataframe(df)
            else:
                st.info(f"No documents found in **{collection_choice}**.")
    except Exception as e:
        st.error(f"Error exploring MongoDB: {e}")
    finally:
        client.close()

load_dotenv()

def main():
    st.title("CSV Uploader & Data Explorer")

    # Use MONGO_URI from .env or fallback to default
    mongo_uri = os.getenv("MONGO_URI", "mongodb://admin:poop1234@mongo:27017/mydatabase?authSource=admin")
    default_db_name = "mydatabase"

    # Choose which page/section to show
    page = st.sidebar.selectbox("Select a page", ["CSV Uploader", "Data Explorer"])

    if page == "CSV Uploader":
        csv_uploader(mongo_uri, default_db_name)
    elif page == "Data Explorer":
        data_explorer(mongo_uri, default_db_name)

if __name__ == "__main__":
    main()
