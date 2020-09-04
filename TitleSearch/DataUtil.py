"""
数据相关工具类
"""


class DataUtil():
    @staticmethod
    def is_null(value):
        if isinstance(value, str):
            if len(value) < 1 or value.lower() in ["nan", "null", "none"]:
                return True
            return False
        if isinstance(value, list):
            if len(value) < 1:
                return True
            return False
        return True
