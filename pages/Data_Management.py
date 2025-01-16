import streamlit as st
import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import plotly.express as px
from datetime import timedelta

load_dotenv()  # Load environment variables from .env

# pages
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

def overview_dashboard(mongo_uri: str, db_name: str):
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]

        # List all collections
        collections = db.list_collection_names()
        if not collections:
            st.info("No collections found in this database.")
            return

        collection_choice = st.selectbox("Select a collection", options=collections, key='overview')
        if collection_choice:
            collection = db[collection_choice]

            latestReadings = get_latest_readings(collection)
            latestReadingsdf = pd.DataFrame(latestReadings)

            st.subheader(f"Latest Readings from Sensors")
            st.dataframe(latestReadingsdf, use_container_width=True)

            parameter = select_box(collection, "ov")

            stats = calculate_stats(collection, parameter)
            statsdf = pd.DataFrame.from_dict(stats, orient="index")

            st.subheader(f"Statistics Table for {parameter}")
            st.dataframe(statsdf, use_container_width=True)

            tvpTable = get_time_vs_parameter(collection, parameter)
            tvpdf = pd.DataFrame(tvpTable)

            # Plot the line graph
            fig = px.line(tvpdf, x="time", y=parameter, color="device", markers=True,
              title="Relationship Between " + parameter + " and Time",
              labels={"time": "Time", parameter: parameter})
            
            st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error exploring MongoDB: {e}")
    finally:
        client.close()

def detailed_data_explorer(mongo_uri: str, db_name: str):
    try:
        client = MongoClient(mongo_uri)
        db = client[db_name]

        # List all collections
        collections = db.list_collection_names()
        if not collections:
            st.info("No collections found in this database.")
            return

        collection_choice = st.selectbox("Select a collection", options=collections, key='detailed')
        if collection_choice:
            collection = db[collection_choice]

            parameter = select_box(collection, "de")

            # detailed graph explorer
            tvpTable = get_time_vs_parameter(collection, parameter)
            tvpdf = pd.DataFrame(tvpTable)

            if not tvpdf.empty:
                latest_timestamp = tvpdf["time"].max()
                earliest_timestamp = tvpdf["time"].min()

                time_range = st.radio(
                    "Select time range:",
                    options=["All time", "Last hour", "Last day", "Last week", "Custom Range"],
                    index=0,
                    help="Choose the time range to filter the data."
                )

                if time_range == "Last hour":
                    start_time = latest_timestamp - 3600
                    tvpdf = tvpdf[(tvpdf["time"] >= start_time) & (tvpdf["time"] <= latest_timestamp)]
                elif time_range == "Last day":
                    start_time = latest_timestamp - 86400
                    tvpdf = tvpdf[(tvpdf["time"] >= start_time) & (tvpdf["time"] <= latest_timestamp)]
                elif time_range == "Last week":
                    start_time = latest_timestamp - 604800
                    tvpdf = tvpdf[(tvpdf["time"] >= start_time) & (tvpdf["time"] <= latest_timestamp)]
                elif time_range == "Custom Range":
                    start_date, end_date = st.slider(
                        "Select custom time range:",
                        min_value=earliest_timestamp,
                        max_value=latest_timestamp,
                        value=(earliest_timestamp, latest_timestamp),
                    )
                    tvpdf = tvpdf[(tvpdf["time"] >= start_date) & (tvpdf["time"] <= end_date)]

                all_devices = tvpdf["device"].unique().tolist()
                selected_devices = st.multiselect(
                    "Select devices to display:",
                    options=all_devices,
                    default=all_devices,
                    help="Choose which devices to include in the graph."
                )

                filtered_df = tvpdf[tvpdf["device"].isin(selected_devices)]

                if not filtered_df.empty:
                    fig = px.line(
                        filtered_df,
                        x="time", 
                        y=parameter,
                        color="device",
                        markers=True,
                        title=f"Relationship Between {parameter} and Time",
                        labels={"time": "Time", parameter: parameter}
                    )
                    st.plotly_chart(fig)
                else:
                    st.warning("No data available for the selected filters.")

            # pagination and table
            st.header("Data Table")

            batch_size = 50 
            total_docs = db[collection_choice].count_documents({})  
            total_pages = (total_docs + batch_size - 1) // batch_size  

            page = st.number_input(
                "Select page", min_value=1, max_value=total_pages, step=1, value=1
            )

            skip_count = (page - 1) * batch_size
            cursor = db[collection_choice].find().skip(skip_count).limit(batch_size)
            docs = list(cursor)

            if docs:
                for doc in docs:
                    doc["_id"] = str(doc["_id"])

                df = pd.DataFrame(docs)
                st.dataframe(df, use_container_width=True)

                st.write(f"Page {page} of {total_pages} (showing {batch_size} documents per page).")
            else:
                st.info(f"No documents found on this page.")

    except Exception as e:
        st.error(f"Error exploring MongoDB: {e}")
    finally:
        client.close()

# helper functions
def select_box(collection, keys):
    sample_document = collection.find_one()

    if sample_document:
        columns = list(sample_document.keys())
        if "_id" in columns:
            columns.remove("_id")
        if "device" in columns:
            columns.remove("device")
        if "ts" in columns:
            columns.remove("ts")

        parameter = st.selectbox("Select an attribute", columns, key=keys)
    else:
        st.warning("The collection is empty or could not fetch a sample document.")
        return None
    
    return parameter

def calculate_stats(collection, parameter):
    devices = collection.distinct("device") 

    stats = {}
    for device in devices:
        readings = list(collection.find({"device": device}, {parameter: 1, "_id": 0}))

        parameterValues = []

        for reading in readings:
            if(parameter in reading):
                parameterValues.append(reading[parameter])

        if parameterValues:
            if(type(parameterValues[0]) is bool) :
                return calculate_frequencies(collection, parameter)
            stats[device] = {
                "min": min(parameterValues),
                "max": max(parameterValues),
                "avg": sum(parameterValues) / len(parameterValues),
            }
        else:
            stats[device] = {"min": None, "max": None, "avg": None}

    return stats

def calculate_frequencies(collection, parameter):
    devices = collection.distinct("device") 

    frequencies = {}
    for device in devices:
        readings = list(collection.find({"device": device}, {parameter: 1, "_id": 0}))

        parameterValues = []

        for reading in readings:
            if(parameter in reading):
                parameterValues.append(reading[parameter])

        if parameterValues:
            total = 0
            count = 0
            for parameterValue in parameterValues:
                if(parameterValue):
                    count = count + 1
                total = total + 1
            
            frequency = count/total
            frequencies[device] = {
                "frequency": frequency
            }
        else:
            frequencies[device] = {"frequency": None}

    return frequencies

def get_latest_readings(collection):
    devices = collection.distinct("device") 

    latest_readings = []
    for device in devices:
        latest_reading = collection.find_one(
            {"device": device},
            sort=[("ts", -1)]
        )
        if latest_reading:
            latest_readings.append(latest_reading)

    return latest_readings

def get_time_vs_parameter(collection, parameter):
    try:
        readings = list(collection.find({parameter: {"$exists": True}}, {"ts": 1, parameter: 1, "device": 1, "_id": 0}))

        tvpTable = [{"time": reading["ts"], "device": reading["device"], parameter: reading[parameter]} for reading in readings]
        tvpTable.sort(key=lambda x: x["time"])

        return tvpTable

    except Exception as e:
        st.error(f"Error fetching time vs parameter data: {e}")
        return []

# main function

def main():
    st.title("Data Ingestion & Exploration")

    # Use MONGO_URI from .env or fallback to default
    mongo_uri = os.getenv("MONGO_URI")
    default_db_name = "mydatabase"

    st.write(
        "Use this page to upload CSVs into MongoDB or to explore existing data. "
        "Check the main page for your other tasks."
    )

    tabs = st.tabs(["Overview Dashboard", "Detailed Data Explorer", "Upload New Data"])

    # Overview Dashboard Tab
    with tabs[0]:
        st.header("Overview Dashboard")
        overview_dashboard(mongo_uri, default_db_name)

    # Detailed Data Explorer Tab
    with tabs[1]:
        st.header("Detailed Data Explorer")
        detailed_data_explorer(mongo_uri, default_db_name)
        # Add filters, data tables, or visualizations here

    # Upload New Data Tab
    with tabs[2]:
        st.header("Upload New Data")
        st.write("Use this tab to upload CSVs into MongoDB")
        csv_uploader(mongo_uri, default_db_name)

# The multipage layout requires you have a "run" statement:
if __name__ == "__main__":
    main()
