import os, csv, requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import mysql.connector

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "lookalike_lens_db")

MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_USER = os.getenv("MILVUS_USER")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")
LOCAL_MILVUS_HOST = "127.0.0.1"
LOCAL_MILVUS_PORT = "19530"

MODEL_NAME = "openai/clip-vit-base-patch32"
VECTOR_DIMENSION = 512
COLLECTION_NAME = "product_vectors"

print("Loading AI model (CLIP)...")
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
print("AI model loaded.")

print("Connecting to Milvus...")
if MILVUS_URI:
    print("Connecting to Zilliz Cloud...")
    connections.connect("default", uri=MILVUS_URI, user=MILVUS_USER, password=MILVUS_PASSWORD)
else:
    print("Connecting to local Milvus...")
    connections.connect("default", host=LOCAL_MILVUS_HOST, port=LOCAL_MILVUS_PORT)
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
    print("Connecting to target database...")
    db_conn = mysql.connector.connect(host=DB_HOST, user=DB_USER, password=DB_PASSWORD, database=DB_NAME)
    cursor = db_conn.cursor()

    cursor.execute("CREATE TABLE IF NOT EXISTS products (product_id INT PRIMARY KEY, name VARCHAR(255), category VARCHAR(100), image_url TEXT)")
    cursor.execute("TRUNCATE TABLE products")
    print("Products table is ready.")

    print("Loading data from products.csv into the database...")
    with open('products.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sql = "INSERT INTO products (product_id, name, category, image_url) VALUES (%s, %s, %s, %s)"
            values = (row['product_id'], row['name'], row['category'], row['image_url'])
            cursor.execute(sql, values)
    db_conn.commit()
    print("Data loaded into database successfully.")

    print("Fetching product data for indexing...")
    cursor.execute("SELECT product_id, image_url FROM products")
    products = cursor.fetchall()
    
    print(f"Found {len(products)} products. Starting vector indexing...")
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
            print(f"Successfully inserted vector for product ID: {product_id}")
        except Exception as e:
            print(f"Error processing product ID {product_id} ({image_url}): {e}")
            
    collection.flush()
    print("\n--- Data insertion complete. Now creating index... ---")

    index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
    collection.create_index(field_name="vector", index_params=index_params)
    print("--- Index created successfully! ---")

except Exception as err:
    print(f"An error occurred: {err}")
finally:
    if 'cursor' in locals():
        cursor.close()
    if 'db_conn' in locals() and db_conn.is_connected():
        db_conn.close()
    connections.disconnect("default")
    print("Database and Milvus connections closed.")