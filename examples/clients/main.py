from face_recognition import HBFace
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# Check if pgvector mode is enabled
use_pgvector = os.getenv("USE_PGVECTOR", "false").lower() == "true"

# Initialize HBFace with or without db_path depending on mode
if use_pgvector:
    # pgvector mode - no pickle file needed
    face_engine_multi = HBFace(
        db_path=None,  # pgvector will be used automatically
        email=os.getenv("SA_EMAIL"),
        password=os.getenv("SA_PASSWORD"),
        client_slug=os.getenv("HB_CLIENTSLUG"),
        api_host=os.getenv("API_HOST"),
    )
else:
    # Legacy pickle mode
    face_engine_multi = HBFace(
        db_path='volumes/src/embeddings/main.pkl',
        email=os.getenv("SA_EMAIL"),
        password=os.getenv("SA_PASSWORD"),
        client_slug=os.getenv("HB_CLIENTSLUG"),
        api_host=os.getenv("API_HOST"),
    )

# Run the face recognition system
face_engine_multi.run()
