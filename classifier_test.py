from classifiers.lr_classifer import LRClassifier
from classifiers.rf_classifier import RFClassifier
from classifiers.xgb_classifier import XgbClassifier
from util.enum.classifier.classifier_type import ClassiferType
import pandas as pd


class ClassifierTest:
    def __init__(self):
        self.__classifier = None
        self.__attr_dict = {}
        self.__classifier_dict = {
            ClassiferType.RandomForest.value: RFClassifier(),
            ClassiferType.LogisticRegression.value: LRClassifier(),
            ClassiferType.XGBoost.value: XgbClassifier()
        }

    def __set_classifer(self, classifier_name):
        self.__classifier = classifier_name

    def __set_attr(self, attr_dict, model_source):
        self.__attr_dict = attr_dict
        self.__attr_dict["checkpoint"] = True
        self.__attr_dict["restore_path"] = model_source

    def execute(self, data_source, model_source, data_sink, classifier_name, attr):
        self.__set_classifer(classifier_name)
        self.__set_attr(attr, model_source)
        self.__classifier.set_attr(self.__attr_dict)
        input_data = pd.read_csv(data_source)
        result = self.__classifier.test(input_data)
        input_data["RESULT"] = result
        output_data = input_data["RESULT"]
        output_data.to_csv(data_sink)