from database import get_connection

def create_tables():
    conn = get_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS facenet.users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255) NOT NULL UNIQUE,
            password VARCHAR(255) NOT NULL,
            email VARCHAR(255) NOT NULL
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS facenet.inference_requests (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255) NOT NULL,
            request_time TIMESTAMP NOT NULL,
            image_path TEXT NOT NULL
        );
        """)
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Tables created successfully.")
    else:
        print("❌ Could not create the tables.")