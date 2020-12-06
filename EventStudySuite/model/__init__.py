# coding=utf-8
import pandas as pd


class DefaultModel(object):
    @staticmethod
    def check_type(name: str, data: object, data_type: object):
        if isinstance(data, data_type):
            pass
        else:
            raise TypeError(f"{name} got wrong type! only accept {data_type}")

    @classmethod
    def real_run(cls, *args, **kwargs) -> pd.Series:
        estimation_df, event_df, stock = args
        cls.check_type('estimation_df', estimation_df, pd.DataFrame)
        cls.check_type('event_df', event_df, pd.DataFrame)
        cls.check_type('stock', stock, str)
        if 'formula' in kwargs.keys():
            cls.check_type('formula', kwargs['formula'], str)
        cls.check_type('factors', kwargs['factors'], list)
        return cls.cal_ar(*args, **kwargs)

    @staticmethod
    def cal_ar(*args, **kwargs):
        raise ValueError('cal_ar have not been defined!')


if __name__ == '__main__':
    pass
