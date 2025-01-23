# db_utils.py
import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from .env if available
load_dotenv()

# Retrieve the MongoDB URI from environment variables
MONGO_URI = os.getenv("MONGO_URI")

# Initialize MongoDB client using MONGO_URI
client = MongoClient(MONGO_URI)

# Access the database specified in the URI or fallback to the environment variable
database_name = os.getenv("MONGO_INITDB_DATABASE", "mydatabase")
db = client[database_name]

def get_collection(collection_name: str):
    """Helper function to retrieve a collection by name."""
    return db[collection_name]

def create_document(data: dict, collection_name: str):
    """
    Insert a new document into the specified MongoDB collection.
    :param data: A dictionary representing the document to insert.
    :param collection_name: The name of the collection to insert into.
    :return: The ID of the inserted document.
    """
    collection = get_collection(collection_name)
    result = collection.insert_one(data)
    return result.inserted_id

def read_document(query: dict, collection_name: str):
    """
    Retrieve a single document matching the query from the specified collection.
    :param query: A dictionary representing the MongoDB query.
    :param collection_name: The name of the collection to search in.
    :return: The document if found, otherwise None.
    """
    collection = get_collection(collection_name)
    return collection.find_one(query)

def read_documents(query: dict, collection_name: str):
    """
    Retrieve multiple documents matching the query from the specified collection.
    :param query: A dictionary representing the MongoDB query.
    :param collection_name: The name of the collection to search in.
    :return: A list of documents.
    """
    collection = get_collection(collection_name)
    return list(collection.find(query))

def update_document(query: dict, new_data: dict, collection_name: str):
    """
    Update a single document in the specified collection that matches the query.
    :param query: A dictionary representing the MongoDB query to match the document.
    :param new_data: A dictionary with the fields to update.
    :param collection_name: The name of the collection to update.
    :return: The number of modified documents.
    """
    collection = get_collection(collection_name)
    result = collection.update_one(query, {"$set": new_data})
    return result.modified_count

def delete_document(query: dict, collection_name: str):
    """
    Delete a single document from the specified collection that matches the query.
    :param query: A dictionary representing the MongoDB query to match the document.
    :param collection_name: The name of the collection to delete from.
    :return: The number of deleted documents.
    """
    collection = get_collection(collection_name)
    result = collection.delete_one(query)
    return result.deleted_count
