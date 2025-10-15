import io, os, requests
from PIL import Image
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel
from pymilvus import connections, Collection
import mysql.connector, mysql.connector.pooling

class Product(BaseModel):
    product_id: int
    name: str
    category: str
    image_url: str
    similarity_score: float

class SearchResponse(BaseModel):
    status: str
    data: List[Product]

app = FastAPI(title="Lookalike Lens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
MODEL_NAME = "openai/clip-vit-base-patch32"

@app.on_event("startup")
def startup_event():
    print("Loading AI model...")
    app.state.model = CLIPModel.from_pretrained(MODEL_NAME)
    app.state.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    print("AI model loaded.")

    print("Connecting to Milvus...")
    if MILVUS_URI:
        print("Connecting to Zilliz Cloud...")
        connections.connect("default", uri=MILVUS_URI, user=MILVUS_USER, password=MILVUS_PASSWORD)
    else:
        print("Connecting to local Milvus...")
        connections.connect("default", host=LOCAL_MILVUS_HOST, port=LOCAL_MILVUS_PORT)
    
    app.state.milvus_collection = Collection(name=COLLECTION_NAME)
    app.state.milvus_collection.load()
    print("Connected to Milvus and collection loaded.")

    print("Creating MySQL/PostgreSQL connection pool...")
    app.state.mysql_pool = mysql.connector.pooling.MySQLConnectionPool(
        pool_name="mysql_pool",
        pool_size=5,
        host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
    )
    print("Database connection pool created.")

def get_image_vector(image: Image.Image):
    inputs = app.state.processor(text=None, images=image, return_tensors="pt", padding=True)
    vector = app.state.model.get_image_features(**inputs).detach().numpy()[0]
    return vector.tolist()

@app.post("/api/v1/search", response_model=SearchResponse)
async def search_for_similar_products(image_file: Optional[UploadFile] = File(None), image_url: Optional[str] = Form(None)):
    image = None
    if image_file:
        contents = await image_file.read()
        image = Image.open(io.BytesIO(contents))
    elif image_url:
        try:
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw)
        except Exception as e:
            return {"status": "error", "message": f"Could not download or open image from URL: {e}"}
    if not image:
        return {"status": "error", "message": "No image or URL provided."}
    query_vector = get_image_vector(image)
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = app.state.milvus_collection.search(data=[query_vector], anns_field="vector", param=search_params, limit=10, output_fields=["product_id"])
    similar_product_ids = [hit.id for hit in results[0]]
    similarity_scores = {hit.id: hit.distance for hit in results[0]}
    if not similar_product_ids:
        return {"status": "success", "data": []}
    try:
        db_conn = app.state.mysql_pool.get_connection()
        cursor = db_conn.cursor(dictionary=True)
        format_strings = ','.join(['%s'] * len(similar_product_ids))
        query = f"SELECT * FROM products WHERE product_id IN ({format_strings})"
        cursor.execute(query, tuple(similar_product_ids))
        products_from_db = cursor.fetchall()
        for product in products_from_db:
            product['similarity_score'] = similarity_scores.get(product['product_id'])
        products_from_db.sort(key=lambda p: p['similarity_score'])
    except Exception as e:
        return {"status": "error", "message": f"Database error: {e}"}
    finally:
        if 'db_conn' in locals() and db_conn.is_connected():
            cursor.close()
            db_conn.close()
    return {"status": "success", "data": products_from_db}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Lookalike Lens API!"}