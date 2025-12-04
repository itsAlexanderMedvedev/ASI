# tests/test_app.py
# Basic test: import predict() and check class_id is valid

from app.predict import predict


def test_predict_returns_valid_class_id():
    # some valid Iris-like input
    features = [5.1, 3.5, 1.4, 0.2]

    result = predict(features)

    # structure checks
    assert "class_id" in result
    assert "class_name" in result

    # type checks
    assert isinstance(result["class_id"], int)
    assert isinstance(result["class_name"], str)

    # value checks: for Iris we expect 0, 1 or 2
    assert result["class_id"] in (0, 1, 2)
    assert result["class_name"] != ""