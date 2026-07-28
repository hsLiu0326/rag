"""Initialize Milvus collection (idempotent)."""

from src.storage.milvus_store import MilvusStore
from src.config import get_settings


def main():
    settings = get_settings()
    store = MilvusStore(uri=settings.milvus_uri, token=settings.milvus_token or None)
    store.ensure_collection(dim=1024)
    print(f"Collection '{store.COLLECTION_NAME}' is ready.")
    store.close()


if __name__ == "__main__":
    main()
