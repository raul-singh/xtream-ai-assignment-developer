import logging
import os
from typing import Any
from dotenv import load_dotenv
import pymongo
from fastapi import Request
import datetime

# Create and initialize logger
logger = logging.getLogger(__name__)

load_dotenv()
DB_URL = os.getenv("DB_URL")
DB_NAME = os.getenv("DB_NAME")
DB_COLLECTION = os.getenv("DB_COLLECTION")


def db_check():
    with pymongo.MongoClient(DB_URL) as client:
        db = client[DB_NAME]
        db_list = client.list_database_names()

        if DB_NAME in db_list:
            logger.info("Database %s found.", DB_NAME)
        else:
            logger.info(
                "Database %s not found. Will create a new one.",
                DB_NAME
            )

        col_list = db.list_collection_names()

        if DB_COLLECTION in col_list:
            logger.info("Collection %s found.", DB_COLLECTION)
        else:
            logger.info(
                "Collection %s not found. Will create a new one.",
                DB_COLLECTION
            )


def store_request(
    request: Request,
    response: Any,
):
    date_time = datetime.datetime.now(tz=datetime.timezone.utc)

    document = {
        "method": request.method,
        "url": str(request.url),
        "query_parameters": dict(request.query_params),
        "response": response,
        "datetime": date_time,
    }

    with pymongo.MongoClient(DB_URL) as client:
        collection = client[DB_NAME][DB_COLLECTION]
        doc_id = collection.insert_one(document)
        logger.info("Inserted API request in database: %s", doc_id)
