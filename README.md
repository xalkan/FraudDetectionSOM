
# Fraud Detection SOM

<img src="https://github.com/xalkan/FraudDetectionSOM/blob/master/output.gif" />



A **self-organizing map** (**SOM**)  is a type of [artificial neural network](https://en.wikipedia.org/wiki/Artificial_neural_network "Artificial neural network") (ANN) that is trained using [unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning "Unsupervised learning") to produce a two-dimensional **map**. I was learning to build one to explore its usage in fraud detection. For more information:
[Self-organizing map on wikipedia](https://en.wikipedia.org/wiki/Self-organizing_map)
[Helpful Medium Article on Fraud Detection using SOMs](https://medium.com/@abhi95.saxena/fraud-detection-using-self-organizing-maps-unsupervised-machine-learning-5c78ae39a584)

### Build and Run
Clone the repo and fire up a terminal in current working directory.
Create a virutual environment

    python -m venv venv
Activate the virtual environment

    # On Linux
    venv/bin/activate
    # On Windows
    venv/bin/activate.bat
Install all the dependencies in requirements.txt

    pip install -r requirements.txt 

Run the self organizing map

    python som.py
