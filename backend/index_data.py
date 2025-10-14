import mysql.connector
import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = ""
DB_NAME = "lookalike_lens_db"
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
MODEL_NAME = "openai/clip-vit-base-patch32"
VECTOR_DIMENSION = 512
COLLECTION_NAME = "product_vectors"

print("Loading AI model (CLIP)...")
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
print("AI model loaded.")

print("Connecting to Milvus...")
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
print("Connected to Milvus.")

if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)
    print(f"Dropped old collection: {COLLECTION_NAME}")

fields = [
    FieldSchema(name="product_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION)
]
schema = CollectionSchema(fields, description="Product image vectors")

collection = Collection(name=COLLECTION_NAME, schema=schema)
print(f"Milvus collection '{COLLECTION_NAME}' created.")

try:
    print("Connecting to MySQL database...")
    mysql_conn = mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME)
    cursor = mysql_conn.cursor()

    print("Fetching product data from MySQL...")
    cursor.execute("SELECT product_id, image_url FROM products")
    products = cursor.fetchall()
    
    print(f"Found {len(products)} products. Starting indexing...")
    
    for product in products:
        product_id, image_url = product
        try:
            image_response = requests.get(image_url, stream=True)
            image_response.raise_for_status()
            image = Image.open(image_response.raw)
            
            inputs = processor(text=None, images=image, return_tensors="pt", padding=True)
            vector = model.get_image_features(**inputs).detach().numpy()[0]
            
            entities = [[product_id], [vector]]
            collection.insert(entities)
            print(f"Successfully inserted product ID: {product_id}")
            
        except Exception as e:
            print(f"Error processing product ID {product_id} ({image_url}): {e}")
            
    collection.flush()
    print("\n--- Data insertion complete. Now creating index... ---")

    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    
    collection.create_index(field_name="vector", index_params=index_params)
    print("--- Index created successfully! ---")

except mysql.connector.Error as err:
    print(f"MySQL Error: {err}")
finally:
    if 'cursor' in locals():
        cursor.close()
    if 'mysql_conn' in locals() and mysql_conn.is_connected():
        mysql_conn.close()
    connections.disconnect("default")
    print("MySQL and Milvus connections closed.")