from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.models import Filter
import uuid


class MemoryManager():
    """
    Chatbot memory manager using qdrant
    """
    def __init__(self, host: str = "localhost", port: int = 6333, timeout: int = 1000) -> None:
        """
        Initialize vector store manager

        Parameters
            host (str): host of qdrant instance
            port (int): port of qdrant instance
            timeout (int): timeout in seconds
        """
        self.client = QdrantClient(host=host, port=port, timeout=timeout)

    def get_collections(self, return_names: bool = True) -> list:
        """
        Get all collections

        Parameters
            return_names (bool): return collection names instead of collection objects

        Returns
            list: list of collections
        """
        collections = self.client.get_collections()
        if return_names:
            return [collection.name for collection in collections.collections]
        return collections.collections
    
    def create_collection(self, name: str = 'long_term_memory', dimension: int = 1536, distance: Distance = Distance.DOT, overwrite: bool = False) -> bool:
        """
        Create a collection

        Parameters
            name (str): name of collection
            dimension (int): dimension of vectors
            distance (Distance): distance function
            overwrite (bool): overwrite collection if it already exists
        """
        if overwrite == False:
            collections = self.get_collections()
            if name in collections:
                return False
        return self.client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dimension, distance=distance)
        )
    
    def get_collection_info(self, name: str = 'long_term_memory') -> dict:
        """
        Get collection info

        Parameters
            name (str): name of collection

        Returns
            dict: collection info
        """
        return self.client.get_collection(name).dict()
    
    def insert_points(self, collection_name: str = 'long_term_memory', points: list = []) -> bool:
        """
        Insert points into collection

        Parameters
            collection_name (str): name of collection
            points (list): list of points to insert (must have a vector and an optional payload)

        Returns
            bool: success
        """
        if len(points) < 1:
            return
        assert all([p.get('vector') is not None for p in points]), "All points must have a vector"

        return self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    payload=p.get('payload') if p.get('payload') is not None else {},
                    vector=p['vector'],
                )
                for p in points
            ],
        )
    
    def search_points(self, collection_name: str = 'long_term_memory', vector: list = [], k: int = 5, append_payload: bool = True, filter: Filter = None) -> list:
        """
        Search points in collection

        Parameters
            collection_name (str): name of collection
            vector (list): vector to search for
            k (int): number of results to return
            append_payload (bool): append payload to results
            filter (Filter): filter results

        Returns
            list: list of results
        """
        return self.client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=k,
            append_payload=append_payload,
            query_filter=filter,
        )
