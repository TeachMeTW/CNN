import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv

# Import the shared db and CRUD functions from your db_utils module
from db_utils.db import db, create_document, read_document, read_documents, update_document, delete_document

# Load environment variables from .env
load_dotenv()

# =============================================================================
# Helper Functions
# =============================================================================
def select_box(collection, key_prefix):
    """Show a selectbox for choosing a parameter from a sample document."""
    sample_document = collection.find_one()
    if sample_document:
        columns = list(sample_document.keys())
        # Remove keys that are less useful for parameter analysis
        for remove_key in ["_id", "device", "ts"]:
            if remove_key in columns:
                columns.remove(remove_key)
        if columns:
            return st.selectbox("Select a parameter for analysis", columns, key=key_prefix)
        else:
            st.warning("No attributes available to select.")
            return None
    else:
        st.warning("The collection is empty or no sample document was found.")
        return None

def calculate_stats(collection, parameter):
    """Calculate basic statistics (min, max, avg) for a given parameter per device."""
    devices = collection.distinct("device")
    stats = {}
    for device in devices:
        readings = list(collection.find({"device": device}, {parameter: 1, "_id": 0}))
        parameter_values = [reading[parameter] for reading in readings if parameter in reading]
        if parameter_values:
            # For booleans, calculate the frequency of True values.
            if isinstance(parameter_values[0], bool):
                return calculate_frequencies(collection, parameter)
            stats[device] = {
                "min": min(parameter_values),
                "max": max(parameter_values),
                "avg": sum(parameter_values) / len(parameter_values),
            }
        else:
            stats[device] = {"min": None, "max": None, "avg": None}
    return stats

def calculate_frequencies(collection, parameter):
    """Calculate the frequency (ratio of True values) for a boolean parameter per device."""
    devices = collection.distinct("device")
    frequencies = {}
    for device in devices:
        readings = list(collection.find({"device": device}, {parameter: 1, "_id": 0}))
        parameter_values = [reading[parameter] for reading in readings if parameter in reading]
        if parameter_values:
            total = len(parameter_values)
            count = sum(1 for val in parameter_values if val)
            frequencies[device] = {"frequency": count / total}
        else:
            frequencies[device] = {"frequency": None}
    return frequencies

def get_latest_readings(collection):
    """Return the latest document (by timestamp) for each device."""
    devices = collection.distinct("device")
    latest_readings = []
    for device in devices:
        latest = collection.find_one({"device": device}, sort=[("ts", -1)])
        if latest:
            latest_readings.append(latest)
    return latest_readings

def get_time_vs_parameter(collection, parameter):
    """Return a sorted list of documents containing time, device, and the specified parameter."""
    try:
        readings = list(
            collection.find(
                {parameter: {"$exists": True}},
                {"ts": 1, parameter: 1, "device": 1, "_id": 0},
            )
        )
        tvp_table = [
            {"time": reading["ts"], "device": reading["device"], parameter: reading[parameter]}
            for reading in readings
        ]
        tvp_table.sort(key=lambda x: x["time"])
        return tvp_table
    except Exception as e:
        st.error(f"Error fetching time vs parameter data: {e}")
        return []

# =============================================================================
# Application Functions
# =============================================================================
def csv_uploader():
    """Upload a CSV and insert its rows into a MongoDB collection."""
    st.subheader("CSV Uploader")
    collections = db.list_collection_names()
    if collections:
        collection_name = st.selectbox("Select a collection for CSV data:", options=collections, key="csv_collection")
    else:
        collection_name = st.text_input("Enter collection name for CSV data:", value="sensor_data", key="csv_collection")
    csv_file = st.file_uploader("Upload CSV", type="csv", key="csv_file")
    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file)
            st.write("CSV Preview:")
            st.dataframe(df.head())
            if st.button("Upload CSV to MongoDB", key="upload_csv"):
                records = df.to_dict(orient="records")
                result = db[collection_name].insert_many(records)
                st.success(f"Inserted {len(result.inserted_ids)} documents into '{collection_name}'.")
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")

def manual_data_editor():
    """
    Show an editable table of documents from a collection so that users can update individual rows,
    add new rows, or mark rows for deletion. A dedicated "Delete" checkbox column is provided.
    Changes are applied using the imported CRUD functions.
    """
    st.subheader("Manual Document Editor")
    collections = db.list_collection_names()
    if collections:
        collection_name = st.selectbox("Select a collection for editing:", options=collections, key="manual_collection")
    else:
        collection_name = st.text_input("Enter collection name for editing:", value="sensor_data", key="manual_collection")
        
    if collection_name:
        try:
            # Retrieve all documents from the collection.
            docs = list(db[collection_name].find())
            if docs:
                # Convert ObjectId values to strings for display/editing.
                for doc in docs:
                    doc["_id"] = str(doc["_id"])
                df_original = pd.DataFrame(docs)
            else:
                df_original = pd.DataFrame(columns=["_id", "device", "ts"])
                st.info("No documents found in the collection. Add new rows below to insert new documents.")

            # Add a "Delete" column to allow marking rows for deletion.
            if "Delete" not in df_original.columns:
                df_original["Delete"] = False

            # Display the editable table.
            edited_df = st.data_editor(df_original, num_rows="dynamic", key="data_editor")

            if st.button("Save Changes", key="save_changes"):
                # Build a dictionary of the original rows keyed by _id.
                original_dict = {}
                for _, row in df_original.iterrows():
                    _id_val = row["_id"]
                    if pd.notna(_id_val) and _id_val != "":
                        original_dict[str(_id_val)] = row.to_dict()

                # Build a set of _id values present in the edited table.
                edited_ids = set()
                for _, row in edited_df.iterrows():
                    _id_val = row.get("_id")
                    if pd.notna(_id_val) and _id_val != "":
                        edited_ids.add(str(_id_val))

                # Identify rows that were physically removed (i.e. deleted from the table).
                missing_ids = set(original_dict.keys()) - edited_ids

                # Process each row in the edited table.
                for _, row in edited_df.iterrows():
                    row_dict = row.to_dict()
                    delete_flag = row_dict.get("Delete", False)
                    _id_val = row_dict.get("_id")
                    if pd.isna(_id_val) or _id_val == "":
                        # New row insertion (if not marked for deletion).
                        if not delete_flag:
                            new_doc = {k: v for k, v in row_dict.items() if k not in ["_id", "Delete"]}
                            inserted_id = create_document(new_doc, collection_name)
                            st.success(f"Inserted new document with id: {inserted_id}")
                    else:
                        _id_str = str(_id_val)
                        if delete_flag:
                            from bson import ObjectId
                            delete_document({"_id": ObjectId(_id_str)}, collection_name)
                            st.success(f"Deleted document with id: {_id_str}")
                        else:
                            # Update only if data has changed.
                            original_row = original_dict.get(_id_str, {})
                            edited_data = {k: row_dict[k] for k in row_dict if k not in ["_id", "Delete"]}
                            original_data = {k: original_row.get(k) for k in row_dict if k not in ["_id", "Delete"]}
                            if edited_data != original_data:
                                from bson import ObjectId
                                update_document({"_id": ObjectId(_id_str)}, edited_data, collection_name)
                                st.success(f"Updated document with id: {_id_str}")

                # Process physical deletions (rows removed from the table).
                for _id_str in missing_ids:
                    from bson import ObjectId
                    delete_document({"_id": ObjectId(_id_str)}, collection_name)
                    st.success(f"Deleted document with id: {_id_str} (row removed)")
        except Exception as e:
            st.error(f"Error reading or updating documents: {e}")

def advanced_collection_management():
    """
    Advanced Collection Management features:
      - Delete a collection.
      - Merge collections.
      - Rename a collection.
    """
    st.subheader("Advanced Collection Management")
    collections = db.list_collection_names()
    if not collections:
        st.info("No collections available.")
        return

    # --- Delete Collection ---
    st.markdown("**Delete Collection**")
    col_to_delete = st.selectbox("Select a collection to delete", options=collections, key="delete_collection")
    # Display the confirmation checkbox outside the button
    confirm = st.checkbox("Confirm deletion", key="delete_confirm")
    if st.button("Delete Collection", key="delete_btn"):
        if confirm:
            db.drop_collection(col_to_delete)
            st.success(f"Collection '{col_to_delete}' deleted.")
        else:
            st.warning("Please confirm deletion by checking the box.")

    st.markdown("---")

    # --- Merge Collections ---
    st.markdown("**Merge Collections**")
    merge_cols = st.multiselect("Select collections to merge", options=collections, key="merge_cols")
    target_collection = st.text_input("Target Collection Name", value="merged_collection", key="merge_target")
    if st.button("Merge Collections", key="merge_btn"):
        if merge_cols:
            merged_docs = []
            for col in merge_cols:
                docs = list(db[col].find())
                for doc in docs:
                    # Remove _id to allow insertion into the target collection.
                    doc.pop("_id", None)
                    merged_docs.append(doc)
            if merged_docs:
                result = db[target_collection].insert_many(merged_docs)
                st.success(f"Merged {len(result.inserted_ids)} documents into '{target_collection}'.")
            else:
                st.info("No documents found in selected collections.")
        else:
            st.warning("Please select at least one collection to merge.")

    st.markdown("---")

    # --- Rename Collection ---
    st.markdown("**Rename Collection**")
    col_to_rename = st.selectbox("Select a collection to rename", options=collections, key="rename_collection")
    new_name = st.text_input("New Collection Name", key="rename_target")
    if st.button("Rename Collection", key="rename_btn"):
        if new_name:
            docs = list(db[col_to_rename].find())
            if docs:
                # Remove _id from each document before inserting into the new collection.
                for doc in docs:
                    doc.pop("_id", None)
                db[new_name].insert_many(docs)
            db.drop_collection(col_to_rename)
            st.success(f"Renamed collection '{col_to_rename}' to '{new_name}'.")
        else:
            st.warning("Please provide a new collection name.")

def combined_data_explorer():
    """
    A unified data explorer that provides:
      - Latest readings per device.
      - Parameter analysis: statistics and time-series plot with time and device filters.
      - A paginated data table.
    """
    st.subheader("Data Explorer")
    collections = db.list_collection_names()
    if not collections:
        st.info("No collections found in the database.")
        return

    collection_choice = st.selectbox("Select a collection", options=collections, key="combined_collection")
    if collection_choice:
        collection = db[collection_choice]

        # --- Latest Readings ---
        st.markdown("#### Latest Readings by Device")
        latest = get_latest_readings(collection)
        if latest:
            st.dataframe(pd.DataFrame(latest), use_container_width=True)
        else:
            st.info("No latest readings found.")

        # --- Parameter Analysis ---
        st.markdown("#### Parameter Analysis")
        parameter = select_box(collection, "combined_param")
        if parameter:
            # Display statistics for the selected parameter.
            stats = calculate_stats(collection, parameter)
            stats_df = pd.DataFrame.from_dict(stats, orient="index")
            st.markdown("**Statistics:**")
            st.dataframe(stats_df, use_container_width=True)

            # Time-series data
            tvp_table = get_time_vs_parameter(collection, parameter)
            if tvp_table:
                tvp_df = pd.DataFrame(tvp_table)
                st.markdown("**Time-Series Analysis:**")
                latest_ts = tvp_df["time"].max()
                earliest_ts = tvp_df["time"].min()

                # Select time range.
                time_range = st.radio(
                    "Select time range:",
                    options=["All time", "Last hour", "Last day", "Last week", "Custom Range"],
                    index=0,
                    key="combined_time_range",
                )
                if time_range == "Last hour":
                    start = latest_ts - 3600
                    tvp_df = tvp_df[(tvp_df["time"] >= start) & (tvp_df["time"] <= latest_ts)]
                elif time_range == "Last day":
                    start = latest_ts - 86400
                    tvp_df = tvp_df[(tvp_df["time"] >= start) & (tvp_df["time"] <= latest_ts)]
                elif time_range == "Last week":
                    start = latest_ts - 604800
                    tvp_df = tvp_df[(tvp_df["time"] >= start) & (tvp_df["time"] <= latest_ts)]
                elif time_range == "Custom Range":
                    start, end = st.slider(
                        "Select custom time range:",
                        min_value=earliest_ts,
                        max_value=latest_ts,
                        value=(earliest_ts, latest_ts),
                        key="combined_custom_range",
                    )
                    tvp_df = tvp_df[(tvp_df["time"] >= start) & (tvp_df["time"] <= end)]

                # Filter by devices.
                all_devices = tvp_df["device"].unique().tolist()
                selected_devices = st.multiselect(
                    "Select devices to display:",
                    options=all_devices,
                    default=all_devices,
                    key="combined_device_select",
                )
                filtered_df = tvp_df[tvp_df["device"].isin(selected_devices)]
                if not filtered_df.empty:
                    fig = px.line(
                        filtered_df,
                        x="time",
                        y=parameter,
                        color="device",
                        markers=True,
                        title=f"Time Series of {parameter}",
                        labels={"time": "Time", parameter: parameter},
                    )
                    st.plotly_chart(fig)
                else:
                    st.info("No data available for the selected filters.")
            else:
                st.info("No time-series data available for this parameter.")

        # --- Paginated Data Table ---
        st.markdown("#### Data Table Viewer")
        batch_size = 50
        total_docs = db[collection_choice].count_documents({})
        total_pages = (total_docs + batch_size - 1) // batch_size
        page = st.number_input("Select page", min_value=1, max_value=total_pages, step=1, value=1, key="combined_page_num")
        skip_count = (page - 1) * batch_size
        cursor = db[collection_choice].find().skip(skip_count).limit(batch_size)
        docs = list(cursor)
        if docs:
            for doc in docs:
                doc["_id"] = str(doc["_id"])
            st.dataframe(pd.DataFrame(docs), use_container_width=True)
            st.write(f"Page {page} of {total_pages} (showing {batch_size} documents per page).")
        else:
            st.info("No documents found on this page.")

# =============================================================================
# Main Application Layout
# =============================================================================
def main():
    st.title("Data Ingestion & Exploration")
    st.write(
        "Manage your MongoDB data by uploading CSVs, editing records directly, and exploring your data with rich statistics and visualizations."
    )

    # Two tabs: one for Upload & Edit Data and one for Data Explorer.
    tabs = st.tabs(["Upload & Edit Data", "Data Explorer"])

    with tabs[0]:
        st.header("Upload & Edit Data")
        csv_uploader()
        st.markdown("---")
        manual_data_editor()
        st.markdown("---")
        advanced_collection_management()

    with tabs[1]:
        st.header("Data Explorer")
        combined_data_explorer()

if __name__ == "__main__":
    main()
