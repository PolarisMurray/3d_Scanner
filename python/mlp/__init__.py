"""Training and evaluation of the voltage-to-distance MLP.

    data.py      load the calibration CSV
    network.py   fit the MLP and save its weights
    train.py     hold-out check, final fit, training plot
    evaluate.py  error of the saved model on every calibration point

Run it with python/train_mlp.py. At scan time the model is read by sensor.SensorModel.
"""
