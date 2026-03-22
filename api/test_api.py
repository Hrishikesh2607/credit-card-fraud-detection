from fastapi.testclient import TestClient
from main import app
import pytest

LEGIT_TXN = {
    "V1":1.19, "V2":0.26, "V3":0.17, "V4":0.45,
    "V5":0.06, "V6":0.59, "V7":0.04, "V8":0.46,
    "V9":0.06, "V10":0.46,"V11":0.83, "V12":0.77,
    "V13":0.20,"V14":0.88,"V15":0.36, "V16":0.19,
    "V17":0.03,"V18":0.08,"V19":0.07, "V20":0.07,
    "V21":0.03,"V22":0.06,"V23":0.01, "V24":0.22,
    "V25":0.24,"V26":0.12,"V27":0.02, "V28":0.01,
    "Amount":25.00, "Time":50000.0
}

@pytest.fixture(scope="session")
def client():
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "threshold" in r.json()

def test_predict_returns_schema(client):
    r = client.post("/predict", json=LEGIT_TXN)
    assert r.status_code == 200
    body = r.json()
    assert "fraud_probability" in body
    assert "flagged" in body
    assert body["risk_tier"] in ["low", "medium", "high"]
    assert 0.0 <= body["fraud_probability"] <= 1.0

def test_zero_amount(client):
    r = client.post("/predict", json={**LEGIT_TXN, "Amount": 0.0})
    assert r.status_code == 200

def test_negative_amount_rejected(client):
    r = client.post("/predict", json={**LEGIT_TXN, "Amount": -10.0})
    assert r.status_code == 422

def test_missing_field_rejected(client):
    txn = {k:v for k,v in LEGIT_TXN.items() if k != "V14"}
    r = client.post("/predict", json=txn)
    assert r.status_code == 422

def test_batch_predict(client):
    r = client.post("/predict/batch",  json={"transactions": [LEGIT_TXN] * 5})
    assert r.status_code == 200
    assert r.json()["total"] == 5

def test_batch_limit(client):
    r = client.post("/predict/batch", json={"transactions": [LEGIT_TXN] * 1001})
    assert r.status_code == 422  
