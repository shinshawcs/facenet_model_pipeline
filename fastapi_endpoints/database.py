import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

#load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

def get_connection():
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        print("✅ Database connection successful")
        return conn
    except Exception as e:
        print("❌ Database connection error:", e)
        return None