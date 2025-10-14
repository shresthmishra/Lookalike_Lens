import csv
import mysql.connector

DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = ""
DB_NAME = "lookalike_lens_db"

try:
    connection = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )
    cursor = connection.cursor()

    with open('products.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sql = "INSERT INTO products (product_id, name, category, image_url) VALUES (%s, %s, %s, %s)"
            values = (row['product_id'], row['name'], row['category'], row['image_url'])
            cursor.execute(sql, values)
            print(f"Inserted product ID: {row['product_id']}")

    connection.commit()
    print("\nData loaded successfully!")

except mysql.connector.Error as err:
    print(f"Error: {err}")

finally:
    if 'connection' in locals() and connection.is_connected():
        cursor.close()
        connection.close()
        print("MySQL connection is closed.")