import pytest
import joblib


def test_linear_regression():
    data = ["Priyanka Chopra and Nick Jonas, who attended the 77th Golden Globe Awards together on Monday morning as presenters surely made a head-turning appearance (but more on that later). This is the story of how Priyanka and Nick, stole our hearts with their PDA on the Golden Globes red carpet."]
    lreg_model = joblib.load("lin-reg.model")
    prediction = lreg_model.predict(data)
    assert prediction[0] == "ENTERTAINMENT"

def test_multinomial_nb():
    data = ["Priyanka Chopra and Nick Jonas, who attended the 77th Golden Globe Awards together on Monday morning as presenters surely made a head-turning appearance (but more on that later). This is the story of how Priyanka and Nick, stole our hearts with their PDA on the Golden Globes red carpet."]
    lreg_model = joblib.load("multi-nb.model")
    prediction = lreg_model.predict(data)
    assert prediction[0] == "ENTERTAINMENT"

def test_linear_svc():
    data = ["Priyanka Chopra and Nick Jonas, who attended the 77th Golden Globe Awards together on Monday morning as presenters surely made a head-turning appearance (but more on that later). This is the story of how Priyanka and Nick, stole our hearts with their PDA on the Golden Globes red carpet."]
    lreg_model = joblib.load("lin-svc.model")
    prediction = lreg_model.predict(data)
    assert prediction[0] == "ENTERTAINMENT"
