import io, os, requests
from PIL import Image
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import ViTImageProcessor, ViTModel
from pymilvus import connections, Collection
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extras import RealDictCursor

class Product(BaseModel):
    product_id: int; name: str; category: str; image_url: str; similarity_score: float

class SearchResponse(BaseModel):
    status: str; data: List[Product]

app = FastAPI(title="Lookalike Lens API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "lookalike_lens_db")
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_USER = os.getenv("MILVUS_USER")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")
LOCAL_MILVUS_HOST = "127.0.0.1"
LOCAL_MILVUS_PORT = "19530"
COLLECTION_NAME = "product_vectors"
MODEL_NAME = "google/vit-base-patch32-224-in21k"

@app.on_event("startup")
def startup_event():
    print("Loading AI model (ViT)..."); app.state.model = ViTModel.from_pretrained(MODEL_NAME); app.state.processor = ViTImageProcessor.from_pretrained(MODEL_NAME); print("AI model loaded.")
    print("Connecting to Milvus...")
    if MILVUS_URI: connections.connect("default", uri=MILVUS_URI, user=MILVUS_USER, password=MILVUS_PASSWORD)
    else: connections.connect("default", host=LOCAL_MILVUS_HOST, port=LOCAL_MILVUS_PORT)
    app.state.milvus_collection = Collection(name=COLLECTION_NAME); app.state.milvus_collection.load(); print("Connected to Milvus.")
    print("Creating PostgreSQL connection pool...")
    dsn = f"dbname='{DB_NAME}' user='{DB_USER}' password='{DB_PASSWORD}' host='{DB_HOST}'"
    app.state.db_pool = SimpleConnectionPool(minconn=1, maxconn=5, dsn=dsn); print("Database pool created.")

def get_image_vector(image: Image.Image):
    inputs = app.state.processor(images=image, return_tensors="pt")
    outputs = app.state.model(**inputs)
    vector = outputs.last_hidden_state.mean(dim=1).detach().numpy()[0]
    return vector.tolist()

@app.post("/api/v1/search", response_model=SearchResponse)
async def search(image_file: Optional[UploadFile]=File(None), image_url: Optional[str]=Form(None)):
    image = None
    if image_file: image = Image.open(io.BytesIO(await image_file.read())).convert("RGB")
    elif image_url:
        try: response = requests.get(image_url, stream=True); response.raise_for_status(); image = Image.open(response.raw).convert("RGB")
        except Exception as e: return {"status": "error", "message": f"Could not open image from URL: {e}"}
    if not image: return {"status": "error", "message": "No image or URL provided."}
    query_vector = get_image_vector(image)
    results = app.state.milvus_collection.search(data=[query_vector], anns_field="vector", param={"metric_type": "L2", "params": {"nprobe": 10}}, limit=10, output_fields=["product_id"])
    ids = [h.id for h in results[0]]; scores = {h.id: h.distance for h in results[0]}
    if not ids: return {"status": "success", "data": []}
    db_conn = None; cursor = None
    try:
        db_conn = app.state.db_pool.getconn()
        cursor = db_conn.cursor(cursor_factory=RealDictCursor)
        q = f"SELECT * FROM products WHERE product_id IN ({','.join(['%s'] * len(ids))})"
        cursor.execute(q, tuple(ids)); products = cursor.fetchall()
        for p in products: p['similarity_score'] = scores.get(p['product_id'])
        products.sort(key=lambda p: p['similarity_score'])
    except Exception as e: return {"status": "error", "message": f"Database error: {e}"}
    finally:
        if cursor: cursor.close()
        if db_conn: app.state.db_pool.putconn(db_conn)
    return {"status": "success", "data": products}

@app.get("/")
def read_root(): return {"message": "Welcome to the Lookalike Lens API!"}