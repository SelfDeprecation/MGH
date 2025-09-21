import pytest

from fastapi.testclient import TestClient

from api.service import app

client = TestClient(app=app)

@pytest.mark.asyncio
def test_api_predict_post():
    data = {'input': ''}
    response = client.post('api/predict', json=data)

    assert response.status_code == 200
    