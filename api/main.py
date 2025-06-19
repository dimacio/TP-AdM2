from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import requests
import jwt
from datetime import datetime, timedelta
from fastapi import Depends, Header
import redis
from datetime import datetime

redis_client = redis.Redis(host="redis", port=6379, db=0, decode_responses=True)


airflow_api_host = "http://airflow-webserver:8080"

app = FastAPI()

SECRET_KEY = "your-very-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

class LoginInput(BaseModel):
    username: str
    password: str


class StockInput(BaseModel):
    open: float
    high: float
    low: float
    volume: float

class DagRunInput(BaseModel):
    dag_id: str


@app.get("/")
def health_check():
    return {"status": "ok"}


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_jwt_token(token: str = Header(...)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        # Optionally check Redis for token revocation
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/login")
def login(username: str, password: str):
    # Authenticate against Airflow using Basic Auth
    url = f"{airflow_api_host}/api/v1/dags"
    try:
        resp = requests.get(url, auth=(username, password), timeout=5)
        print(f"Airflow response: {resp.status_code} - {resp.text}")
        if resp.status_code != 200:
           raise HTTPException(status_code=resp.status_code, detail=f"Airflow API error: {resp.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Airflow authentication failed: {str(e)}")

    # Store password in Redis with TTL
    redis_client.setex(f"airflow:{username}:password", 3600, password)
    access_token = create_access_token(data={"sub": username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/trigger-new-dag-run")
def trigger_new_dag_run(
    data: DagRunInput,
    username: str = Depends(verify_jwt_token)
):
    password = redis_client.get(f"airflow:{username}:password")
    if not password:
        raise HTTPException(status_code=401, detail="Session expired, please log in again.")
    
    url = f"{airflow_api_host}/api/v1/dags/{data.dag_id}/dagRuns"
    payload = {}

    try:
        resp = requests.post(url, json=payload, auth=(username, password))
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as e:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Airflow API error: {resp.text}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/predict")
def predict(data: StockInput):
    return {"prediction": 42}  # Dummy response for testing
