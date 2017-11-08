from classifiers.classifier import Classifier
from sklearn.ensemble import RandomForestClassifier


class RFClassifier(Classifier):
    def __init__(self):
        Classifier.__init__(self)
        self._clf = RandomForestClassifier()