import io
import json
import numpy as np
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
from fastapi import FastAPI, UploadFile, Form,File, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from google.cloud import storage
from uuid import uuid4
import psycopg2
import requests
from database import get_connection
from models import create_tables
from dotenv import load_dotenv
import os
from prometheus_fastapi_instrumentator import Instrumentator

#load_dotenv()
TRITON_URL = os.getenv("TRITON_URL")
#triton_client = httpclient.InferenceServerClient(url=TRITON_URL)
triton_client = grpcclient.InferenceServerClient(url=TRITON_URL,verbose=True)
#url="http://triton-service.airflow.svc.cluster.local:8000"

create_tables()

app = FastAPI()
Instrumentator().instrument(app).expose(app)

BUCKET_NAME = "my-facenet-bucket"
storage_client = storage.Client()

class User(BaseModel):
    username: str
    password: str
    email: str


@app.post("/register")
def register(user: User):
    conn = get_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection error")

    try:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO facenet.users (username, password, email)
        VALUES (%s, %s, %s) RETURNING id;
        """, (user.username, user.password, user.email))
        
        user_id = cursor.fetchone()['id']
        conn.commit()
        cursor.close()
        conn.close()
        return {"status": "success", "user_id": user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def upload_to_gcp(file: UploadFile):
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"images/{uuid4()}-{file.filename}")
    blob.upload_from_file(file.file)
    blob.make_public()
    return blob.public_url

def triton_infer(image_data: bytes):
    try:
        img_np = np.frombuffer(image_data, dtype=np.uint8).reshape(1, -1)

        #input_tensor = httpclient.InferInput("image", img_np.shape, "UINT8")
        input_tensor = grpcclient.InferInput("image", img_np.shape, "UINT8")
        input_tensor.set_data_from_numpy(img_np)

        #output_tensor = httpclient.InferRequestedOutput("category")
        output_tensor = grpcclient.InferRequestedOutput("category")
        response = triton_client.infer("ensemble_pipeline", inputs=[input_tensor], outputs=[output_tensor])

        result = response.as_numpy("category")
        if result is not None:
            pred_label = int(result[0])
            with open("label_map.json", "r") as f:
                label_map = json.load(f)
            pred_name = label_map.get(str(pred_label), "Unknown")
            return pred_name
        else:
            return "Prediction failed"
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/inference")
async def create_inference_request(username: str = Form(...), file: UploadFile = File(...)):

    try:
        image_url = upload_to_gcp(file)
       
        conn = get_connection()
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO facenet.inference_requests (username, request_time, image_path)
        VALUES (%s, %s, %s) RETURNING id;
        """
        cursor.execute(insert_query, (username, datetime.now(), image_url))
        conn.commit()
        request_id = cursor.fetchone()['id']

        cursor.close()
        conn.close()
        
        file.file.seek(0) 
        image_data = file.file.read()
        prediction = triton_infer(image_data)
        return JSONResponse({
            "status": "success",
            "request_id": request_id,
            "username": username,
            "image_url": image_url,
            "prediction": prediction
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))