from fastapi import FastAPI

from api import schemas, helpers

app = FastAPI(title='X5 entity allocator')

@app.post('/api/predict')
async def api_predict_post(request:schemas.PredictRequestModel = None):
    if not request:
        return ''
    response_item = _create_response(request)
    return [response_item]


def _create_response(request:schemas.PredictRequestModel) -> schemas.PredictResponseItem:

    #response_array = helpers.get_predicted_annotations(request.input)
    response_array = [(0, 'B-TYPE'), (1, 'B-PERCENT'), (2, 'B-VOLUME')]

    return [schemas.PredictResponseItem(
            start_index=i, 
            end_index=0, 
            entity=t
        ).model_dump_json() 
        for i,t in response_array]