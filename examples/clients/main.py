from face_recognition import HBFace
import os
from dotenv import load_dotenv

load_dotenv(override=True)

face_engine_multi = HBFace(
    db_path='volumes/src/embeddings/main.pkl',  # will be removed
    email=os.getenv("SA_EMAIL"),
    password=os.getenv("SA_PASSWORD"),
    client_slug=os.getenv("HB_CLIENTSLUG"),
    api_host=os.getenv("API_HOST"),
)

# Run the face recognition system
face_engine_multi.run()
