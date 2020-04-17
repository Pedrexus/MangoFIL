from mango.models import get_model, LeNet5

def test_get_model():
    assert get_model('LeNet5') == LeNet5
