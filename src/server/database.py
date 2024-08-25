import datetime
import logging
import os
import sys
from typing import Any

import pymongo
from fastapi import Request
from pymongo.errors import ServerSelectionTimeoutError

# Create and initialize logger
logger = logging.getLogger(__name__)


def db_check():

    db_url = os.getenv("DB_URL")
    db_name = os.getenv("DB_NAME")
    db_collection = os.getenv("DB_COLLECTION")

    try:
        with pymongo.MongoClient(db_url, connectTimeoutMS=10000) as client:
            db = client[db_name]
            db_list = client.list_database_names()

            if db_name in db_list:
                logger.info("Database %s found.", db_name)
            else:
                logger.info(
                    "Database %s not found. Will create a new one.",
                    db_name
                )

            col_list = db.list_collection_names()

            if db_collection in col_list:
                logger.info("Collection %s found.", db_collection)
            else:
                logger.info(
                    "Collection %s not found. Will create a new one.",
                    db_collection
                )

    except ServerSelectionTimeoutError:
        logger.error(
            "Cannot establish connection with database. "
            "A MongoDB instance is required for the server to run."
        )
        sys.exit(1)


def store_request(
    request: Request,
    response: Any,
):

    db_url = os.getenv("DB_URL")
    db_name = os.getenv("DB_NAME")
    db_collection = os.getenv("DB_COLLECTION")

    date_time = datetime.datetime.now(tz=datetime.timezone.utc)

    document = {
        "method": request.method,
        "url": str(request.url),
        "path": str(request.url.path),
        "query_parameters": dict(request.query_params),
        "response": response,
        "datetime": date_time,
    }

    with pymongo.MongoClient(db_url) as client:
        collection = client[db_name][db_collection]
        doc_id = collection.insert_one(document)
        logger.info("Inserted API request in database: %s", doc_id)
