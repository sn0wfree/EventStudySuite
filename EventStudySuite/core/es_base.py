# coding=utf-8
import warnings
from collections import namedtuple
from functools import partial

import numpy as np
import pandas as pd
from scipy.stats import t

from EventStudySuite.error import EventDataKeyError
from EventStudySuite.model import DefaultModel
from EventStudySuite.utils import boost_up, cached_property

event_tuple = namedtuple('event', ('before_event_window', 'event_window', 'post_event_window', 'gap'))


class EventStudyUtils(object):
    @staticmethod
    def split_event(return_df: pd.DataFrame, event_happen_day: str, stock, date='Date', factors=['Mkt_RF'],
                    event_info: tuple = (250, 20, 10, 1), detect=True):
        if isinstance(stock, str):
            variables = [date, stock] + factors
        elif isinstance(stock, list):
            variables = [date] + stock + factors
        else:
            raise TypeError('stock got wrong type! only accept str or list')
        if detect:
            if event_info[-2] < 0:
                if event_info[-1] > abs(event_info[-2]) + 1:
                    pass
                else:
                    warnings.warn('estimation period covered event date! will shift estimation period!')
                    event_info = (event_info[0], event_info[1], event_info[2], abs(event_info[-2]) + 1)

        event_set = event_tuple(*event_info)
        post_event_window = event_set.post_event_window
        after_event_window = event_set.event_window - post_event_window
        try:
            data = return_df[variables]
        except KeyError as e:
            raise EventDataKeyError(e)
        event_index_loc = int(
            (pd.to_datetime(data[date]) - pd.to_datetime(event_happen_day)).abs().sort_values().index[0])
        estimation_df = data.loc[event_index_loc - (
                event_set.before_event_window + event_set.post_event_window + event_set.gap): event_index_loc - (
                event_set.post_event_window + event_set.gap), variables]
        event_df = data.loc[event_index_loc - event_set.post_event_window: event_index_loc + after_event_window,
                   variables].reset_index(drop=True)

        event_df.index = event_df.index - event_set.post_event_window

        return estimation_df, event_df

    @staticmethod
    def cal_ar_multi_no_split_event(estimation_df, event_df, stock: str, factors=['Mkt_RF'],
                                    ar_only=True, model=DefaultModel):
        if issubclass(model, DefaultModel):
            ar_series = model.real_run(estimation_df, event_df, stock, factors=factors)
        else:
            raise TypeError(f'model got wrong definition: {model.__name__}! should be a DefaultModel-liked class')
        if ar_only:
            # output_cols = [f'{stock}_ar']
            return pd.DataFrame(ar_series)
        else:
            event_df[f'{stock}_ar'] = ar_series
            return event_df

    @classmethod
    def cal_ar_single(cls, return_df: pd.DataFrame, event_happen_day: str, stock: str,
                      date='Date', factors=['Mkt_RF'], event_info: tuple = (250, 20, 10, 1), ar_only=True,
                      model=DefaultModel):

        estimation_df, event_df = cls.split_event(return_df, event_happen_day, stock, date=date,
                                                  event_info=event_info, factors=factors)

        return cls.cal_ar_multi_no_split_event(estimation_df, event_df, model, stock, factors=factors,
                                               ar_only=ar_only, )

    @classmethod
    def group_split_event(cls, return_df: pd.DataFrame, event_dict: dict, date='Date',
                          factors=['Mkt_RF'],
                          event_info: tuple = (250, 20, 10, 1)):
        sub_tasks = {}
        for stock, event_happen_day in event_dict.items():
            if event_happen_day in sub_tasks.keys():
                sub_tasks[event_happen_day].append(stock)
            else:
                sub_tasks[event_happen_day] = [stock]
        for event_happen_day, stock_list in sub_tasks.items():
            group_estimation_df, group_event_df = cls.split_event(return_df, event_happen_day, stock_list, date=date,
                                                                  event_info=event_info, factors=factors)
            for stock in stock_list:
                yield group_estimation_df, group_event_df, stock

    @classmethod
    def cal_residual(cls, return_df: pd.DataFrame, event_dict: dict, date='Date',
                     factors=['Mkt_RF'], model=DefaultModel,
                     event_info: tuple = (250, 20, 10, 1), ar_only=True, boost=True):

        """
        计算异常收益率并加总异常收益率(CAR)
[公式] 计算的是股票 [公式] 在第 [公式] 天的异常收益率，为了研究事件对整体证券定价的影响，还需要计算平均异常收益率 [公式] 和累积异常收益率 [公式] 。
通常而言，平均异常收益率是针对某一时点、对所有公司的异常收益率求平均，计算方式如下所示：

[公式]

        :param return_df:
        :param event_dict:
        :param date:
        :param factors:
        :param model:
        :param event_info:
        :param ar_only:
        :param boost:
        :return:
        """
        # func = partial(cls.cal_ar_single, date=date, factors=factors, model=model,
        #                event_info=event_info, ar_only=ar_only)
        func = partial(cls.cal_ar_multi_no_split_event, factors=factors, model=model, ar_only=ar_only)

        # tasks = ((return_df, event_happen_day, stock) for stock, event_happen_day in event_dict.items())
        tasks = cls.group_split_event(return_df, event_dict, date=date, factors=factors, event_info=event_info)

        if boost:
            holder = boost_up(func, tasks, star=True)
        else:
            holder = [func(group_estimation_df, group_event_df, stock) for group_estimation_df, group_event_df, stock in
                      tasks]
        return pd.concat(holder, axis=1)


class EventStudy(EventStudyUtils):
    def __call__(self, return_df: pd.DataFrame, event_dict: dict, date='Date', factors=['Mkt_RF'],
                 event_info: tuple = (250, 20, 10, 1), ):
        data, new_event_dict = self.detect_multi_event_point(return_df, event_dict)
        self.data = data
        self.event_dict = new_event_dict
        self.event_info = event_info
        self.cols = dict(date=date, factors=factors)
        return self.result

    def __init__(self, return_df: (None, pd.DataFrame) = None, event_dict: (dict, None) = None, date='Date',
                 factors=['Mkt_RF'], model=DefaultModel, event_info: tuple = (250, 20, 10, 1), ar_only=True,
                 boost=False):
        """
        >>>  es = EventStudy(return_df,event_dict=event_dict,date='Date', factors=['Mkt_RF'],
                             formula="{stock} ~ 1 + {factors_str}",event_info=(250, 10, 5, 1))

        >>> es.result
                      ar    var_ar       car   var_car t_statistic  p_values
            -5 -0.005464  0.000102 -0.005464  0.000102 -0.540883  0.294536
            -4 -0.002986  0.000102 -0.008450  0.000204 -0.591467  0.277372
            -3 -0.004987  0.000102 -0.013437  0.000306 -0.767973  0.221615
            -2 -0.011591  0.000102 -0.025028  0.000408 -1.238760  0.108300
            -1 -0.006547  0.000102 -0.031576  0.000510 -1.397834  0.081703
            0  -0.020056  0.000102 -0.051632  0.000612 -2.086547  0.018973
            1  -0.005131  0.000102 -0.056763  0.000714 -2.123751  0.017339
            2  -0.027366  0.000102 -0.084129  0.000816 -2.944349  0.001771
            3   0.004946  0.000102 -0.079183  0.000918 -2.612757  0.004764
            4   0.007692  0.000102 -0.071491  0.001021 -2.237908  0.013056
            5  -0.001437  0.000102 -0.072928  0.001123 -2.176649  0.015223




        :param return_df:
        :param event_dict:
        :param date:

        :param factors:

        :param event_info:
        :param ar_only:
        :param boost:
        """

        data, new_event_dict = self.detect_multi_event_point(return_df, event_dict)
        self.data = data
        self.event_dict = new_event_dict
        self.cols = dict(date=date, factors=factors)
        self.model = model
        self.event_info = event_info
        self.ar_only = ar_only
        self.boost = boost

    @staticmethod
    def detect_multi_event_point(return_df, event_dict: dict):
        if return_df is None:
            return None, None
        cols = return_df.columns.tolist()
        new_event_dict = {}
        for k, v in event_dict.items():
            if k not in cols:
                raise ValueError(f'{k} not found')
            data_series = return_df[k]
            if isinstance(v, str):
                new_k = k + f"_{pd.to_datetime(v).strftime('%Y%m%d')}"
                new_event_dict[new_k] = v
                return_df[new_k] = data_series
            elif isinstance(v, (list, tuple)):
                v = list(set(v))

                if len(v) >= 1:
                    # cols = return_df.columns.tolist()
                    # new_event_dict[k] = v
                    new_dt_k_dict = {k + f"_{pd.to_datetime(dt).strftime('%Y%m%d')}": dt for dt in v}
                    # for dt in v:
                    #     new_k = k + f"_{pd.to_datetime(dt).strftime('%Y%m%d')}"
                    extra_return_df = pd.DataFrame([data_series.values.ravel() for _ in v],
                                                   index=list(new_dt_k_dict.keys()), columns=data_series.index).T
                    return_df = pd.concat([return_df, extra_return_df], axis=1)
                    # return_df = return_df.reindex(columns=cols +list(new_dt_k_dict.keys()) ,fill_value=)

                    new_event_dict.update(new_dt_k_dict)
                else:
                    raise ValueError(f'event list is empty for {k}')
            else:
                raise ValueError('got wrong event list type')
        return return_df, new_event_dict

    @property
    def arr(self):
        return self.ar

    @property
    def residual(self):
        if self.data is None:
            raise TypeError('return_df is not setup!')
        if self.event_dict is None:
            raise TypeError('event_dict is not setup!')
        if self.event_dict is None:
            raise TypeError('date and factors are not setup!')
        return self.cal_residual(self.data, self.event_dict, date=self.cols['date'],
                                 factors=self.cols['factors'], model=self.model,
                                 event_info=self.event_info, ar_only=self.ar_only, boost=self.boost)

    @cached_property
    def ar(self):
        """
            Cross-sectional aggregation
         the cross-sectional mean abnormal return for any period t


        :return:
        """
        return self.residual.mean(axis=1, skipna=True)

    @cached_property
    def std_ar(self):
        return self.ar.std()

    @cached_property
    def var_ar(self):
        return self.ar.var()

    @cached_property
    def t_stats(self):
        return self.car.squeeze() / np.sqrt(self.var_car)

    @cached_property
    def p_value(self):
        return 1.0 - t.cdf(abs(self.t_stats), event_tuple(*self.event_info).before_event_window - 1)

    @cached_property
    def var_car(self):
        """

        σ , = Lσ(AR)

        :return:
        """

        return [(i * var) for i, var in enumerate([self.var_ar] * self.ar.shape[0], 1)]

    #
    @cached_property
    def car(self):
        """

        The cumulative average residual method (CAR) uses as the abnormal performance measure
        the sum of each month’s average abnormal performance

        :return:
        """
        return pd.DataFrame(self.ar).cumsum()

    @cached_property
    def result(self):
        p = self.p_value
        t_statistic = self.t_stats.tolist()
        ar = self.ar.tolist()
        var_ar = [self.var_ar] * len(ar)
        car = self.car.squeeze()
        var_car = self.var_car
        return pd.DataFrame([ar, var_ar, car, var_car, t_statistic, p], columns=car.index,
                            index=['ar', 'var_ar', 'car', 'var_car', 't_stats', 'p_values']).T


if __name__ == '__main__':
    pass
