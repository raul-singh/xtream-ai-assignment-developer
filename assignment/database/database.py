import logging
from typing import Any
import pymongo
from fastapi import Request
import datetime

# Create and initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    encoding='utf-8',
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    level=logging.INFO
)


def db_check(
    client_url: str = "mongodb://localhost:27017/",
    db_name: str = "diamond_db",
    col_name: str = "api_requests"
):
    with pymongo.MongoClient(client_url) as client:
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

        if col_name in col_list:
            logger.info("Collection %s found.", col_name)
        else:
            logger.info(
                "Collection %s not found. Will create a new one.",
                col_name
            )


def store_request(
    request: Request,
    response: Any,
    client_url: str = "mongodb://localhost:27017/",
    db_name: str = "diamond_db",
    col_name: str = "api_requests"
):
    date_time = datetime.datetime.now(tz=datetime.timezone.utc)

    document = {
        "method": request.method,
        "url": str(request.url),
        "query_parameters": dict(request.query_params),
        "response": response,
        "datetime": date_time,
    }

    with pymongo.MongoClient(client_url) as client:
        collection = client[db_name][col_name]
        doc_id = collection.insert_one(document)
        logger.info("Inserted API request in database: %s", doc_id)
