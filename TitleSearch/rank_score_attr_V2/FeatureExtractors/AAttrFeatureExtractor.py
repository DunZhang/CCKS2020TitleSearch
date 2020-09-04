from abc import ABCMeta, abstractmethod
from typing import List, Union


class AAttrFeatureExtractor(metaclass=ABCMeta):
    @abstractmethod
    def extract_features(self, title: str, attr_values: List[str], **kwargs) -> List[List[float]]:
        pass

    @abstractmethod
    def title_contain_attr(self, title: str, *args, **kwargs) -> Union[None, str]:
        pass

    @abstractmethod
    def get_num_features(self) -> int:
        pass
