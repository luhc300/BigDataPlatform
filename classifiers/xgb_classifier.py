from xgboost.sklearn import XGBClassifier
from classifiers.classifier import Classifier

class XgbClassifier(Classifier):
    def __init__(self):
        Classifier.__init__(self)
        self._clf = XGBClassifier()