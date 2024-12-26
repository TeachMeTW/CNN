import streamlit as st
import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

def csv_uploader(mongo_uri: str, db_name: str):
    """
    Allows the user to upload a CSV and store it into a MongoDB collection.
    """
    st.subheader("CSV Uploader")

    collection_name = st.text_input(
        "Collection Name to insert CSV data into:", 
        value="sensor_data"
    )

    # Upload CSV
    csv_file = st.file_uploader("Upload CSV", type="csv")
    if csv_file is not None:
        # Show CSV preview
        df = pd.read_csv(csv_file)
        st.write("CSV Preview:")
        st.dataframe(df.head())

        if st.button("Upload to MongoDB"):
            try:
                client = MongoClient(mongo_uri)
                db = client[db_name]

                # Convert DataFrame to list of dicts
                records = df.to_dict(orient="records")

                # Insert into the chosen collection
                result = db[collection_name].insert_many(records)
                st.success(
                    f"Inserted {len(result.inserted_ids)} documents into '{collection_name}'."
                )
            except Exception as e:
                st.error(f"Error uploading to MongoDB: {e}")
            finally:
                client.close()

def data_explorer(mongo_uri: str, db_name: str):
    """
    Explore data from the chosen MongoDB database and collection.
    """
    st.subheader("Data Explorer")
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
            # Limit to 100 docs for performance (example logic)
            cursor = db[collection_choice].find().limit(100)
            docs = list(cursor)
            st.write(
                f"Showing first 100 documents from **{collection_choice}** "
                "(if that many exist)."
            )
            st.write(
                "Total documents in collection may be more. "
                "Use your own queries for deeper analysis."
            )

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

def main():
    st.title("Data Ingestion & Exploration")

    # Use MONGO_URI from .env or fallback to default
    mongo_uri = os.getenv("MONGO_URI")
    default_db_name = "mydatabase"

    st.write(
        "Use this page to upload CSVs into MongoDB or to explore existing data. "
        "Check the main page for your other tasks."
    )

    # Create two expandable sections
    with st.expander("Upload CSV to MongoDB"):
        csv_uploader(mongo_uri, default_db_name)

    with st.expander("Explore Existing Data in MongoDB"):
        data_explorer(mongo_uri, default_db_name)

# The multipage layout requires you have a "run" statement:
if __name__ == "__main__":
    main()
