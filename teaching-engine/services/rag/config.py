"""
Configuration for RAG service.
"""

# Trumpet Herald Callet Forum URL
# Note: The 'sid' (session ID) parameter may expire - get a fresh one by logging into the forum
TRUMPET_HERALD_CALLET_FORUM_URL = "https://www.trumpetherald.com/forum/viewforum.php?f=16"

# Forum information
FORUM_INFO = {
    "name": "Trumpet Herald - Jerome Callet Forum",
    "forum_id": 16,
    "moderator": {
        "username": "tptguy",
        "name": "Kyle Schmeer",
        "note": "Studied with Jerry for a long time. Owner of the Callet Sima and New York Soloist trumpet business."
    },
    "user_usernames": [
        "trumpetlogic",  # Current username
        "mythopoeic"     # Older posts
    ]
}

# Default ingestion settings
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.3

# Embedding model
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Vector store
DEFAULT_COLLECTION_NAME = "callet_knowledge"

