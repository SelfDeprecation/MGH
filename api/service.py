from fastapi import FastAPI

from api import schemas

app = FastAPI(title='X5 entity allocator')

@app.post('/api/predict')
async def api_predict_post(request:schemas.PredictRequestModel = None):
    if not request:
        return ''
    response_item = _create_response(request)
    return [response_item]


def _create_response(request:schemas.PredictRequestModel) -> schemas.PredictResponseItem:

    pass

    return schemas.PredictResponseItem(
            start_index=0, 
            end_index=0, 
            entity=request.input
        ).model_dump_json()