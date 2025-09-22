import typing

from src import predict


def get_predicted_annotations(request: str) -> typing.List[typing.Tuple[int, str]]:
    return predict.predict(request)


def _add_indexes(sentence: str, annotations: typing.List[typing.Tuple[int, str]]) -> typing.List[typing.Tuple[int, int, str]]:
    pass