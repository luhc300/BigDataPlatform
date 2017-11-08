from processors.down_sample_processor import DownSampleProcessor
from processors.model_feature_selection_processor import ModelFeatureSelectionProcessor
from processors.normalize_processor import NormalizeProcessor
from util.enum.preprocessor.processor_type import ProcessorType
import pandas as pd

class DataPreprocess:
    def __init__(self):
        self.__processor = None
        self.__attr_dict = {}
        self.__processor_dict = {
            ProcessorType.Normalize.value: NormalizeProcessor(),
            ProcessorType.DownSample.value: DownSampleProcessor(),
            ProcessorType.ModelFeatureSelection.value: ModelFeatureSelectionProcessor()
        }

    def __set_processor(self, processor_name : str):
        self.__processor = self.__processor_dict.get(processor_name)

    def __set_attr(self, attr_dict : dict):
        self.__attr_dict = attr_dict

    def execute(self, data_source : str, data_sink : str, processor_name : str, attr : dict):
        self.__set_processor(processor_name)
        self.__set_attr(attr)
        self.__processor.set_attr(self.__attr_dict)
        input_data = pd.read_csv(data_source)
        output_data = self.__processor.execute(input_data)
        output_data.to_csv(data_sink)
