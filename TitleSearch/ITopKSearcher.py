from abc import ABCMeta, abstractmethod
from typing import List


class ITopKSearcher(metaclass=ABCMeta):
    @abstractmethod
    def get_topk_ent_ids(self, title=str, topk: int = 100) -> List[str]:
        pass
