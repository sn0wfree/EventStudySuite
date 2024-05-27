# coding=utf-8

import numpy as np
import pandas as pd
import statsmodels.api as sm

from EventStudySuite.model.DefaultModel import DefaultModel


class MarketModelFast(DefaultModel):
    @staticmethod
    def cal_ar(estimation_df, event_df, stock: str, formula: str = "{stock} ~ 1 + {factors}", factors=['Mkt_RF'],
               raise_error=False):
        """
市场模型（Market Model，简写 MM）是一个线性模型，通过个股收益率对某个宽基指数收益
率的回归来确定个股的预期收益，
        :param raise_error:
        :param estimation_df:  pd.DataFrame
        :param event_df:  pd.DataFrame
        :param stock:
        :param formula:
        :param factors:
        :return:
        """
        # f1 = formula.format(stock=stock, factors="+".join(factors))
        y_array = estimation_df[stock]
        w = 1 - y_array.isna() * 1
        length = estimation_df.shape[0]
        if w.sum() == length:
            y_array = y_array.values.ravel()
            x_array = estimation_df[factors[0]].values.ravel()
            sum_y = np.sum(y_array)
            sum_x = np.sum(x_array)
            sum_xy = np.dot(x_array, y_array.T)
            sum_x2 = np.dot(x_array, x_array)
            beta_Mkt = ((sum_x * sum_y) / length - sum_xy) / ((sum_x * sum_x) / length - sum_x2)
            alpha = (sum_y - sum_x * beta_Mkt) / length
            event_df[f'{stock}_er'] = np.dot(event_df[factors].values, beta_Mkt) + alpha
            # event_df[f'{stock}_ar'] = event_df.eval(f'{stock} - {stock}_er')
            series = event_df.eval(f'{stock} - {stock}_er')
            series.index = event_df.index
            return series
        else:
            series = pd.Series([None] * event_df.shape[0])
            series.index = event_df.index
            return series


class MarketModel(DefaultModel):
    @staticmethod
    def cal_ar(estimation_df, event_df, stock: str, formula: str = "{stock} ~ 1 + {factors}", factors=['Mkt_RF'],
               raise_error=False):
        """
市场模型（Market Model，简写 MM）是一个线性模型，通过个股收益率对某个宽基指数收益
率的回归来确定个股的预期收益，
        :param raise_error:
        :param estimation_df:  pd.DataFrame
        :param event_df:  pd.DataFrame
        :param stock:
        :param formula:
        :param factors:
        :return:
        """
        f1 = formula.format(stock=stock, factors="+".join(factors))
        w = 1 - estimation_df[stock].isna() * 1
        length = estimation_df.shape[0]
        if w.sum() == length:

            # print(f1)
            if estimation_df.dropna().shape[0] == estimation_df.shape[0]:
                models = sm.OLS.from_formula(f1, data=estimation_df).fit()
                params = models.params
                beta_Mkt = params[factors]
                alpha = params["Intercept"]
                bse = models.bse
                event_df[f'{stock}_er'] = np.dot(event_df[factors].values, beta_Mkt) + alpha
                # event_df[f'{stock}_ar'] = event_df.eval(f'{stock} - {stock}_er')
                series = event_df.eval(f'{stock} - {stock}_er')
                series.index = event_df.index
                return series
            else:
                if raise_error:
                    raise ValueError('estimation_df has nan data!')
                else:

                    series = pd.Series([None] * event_df.shape[0])
                    series.index = event_df.index
                    return series
        else:
            series = pd.Series([None] * event_df.shape[0])
            series.index = event_df.index
            return series


if __name__ == '__main__':
    pass
