# coding=utf-8

import numpy as np
import pandas as pd
import statsmodels.api as sm

from EventStudySuite.model import DefaultModel


class MarketModel(DefaultModel):
    @staticmethod
    def cal_ar(estimation_df, event_df, stock: str, formula: str = "{stock} ~ 1 + {factors}", factors=['Mkt_RF'],
               raise_error=False):
        """

        :param estimation_df:  pd.DataFrame
        :param event_df:  pd.DataFrame
        :param stock:
        :param formula:
        :param factors:
        :return:
        """
        f1 = formula.format(stock=stock, factors="+".join(factors))
        # print(f1)
        if estimation_df.dropna().shape[0] == estimation_df.shape[0]:
            models = sm.OLS.from_formula(f1, data=estimation_df).fit()
            params = models.params
            beta_Mkt = params[factors]
            alpha = params["Intercept"]
            bse = models.bse
            event_df[f'{stock}_er'] = np.dot(event_df[factors].values, beta_Mkt) + alpha
            # event_df[f'{stock}_ar'] = event_df.eval(f'{stock} - {stock}_er')
            return event_df.eval(f'{stock} - {stock}_er')
        else:
            if raise_error:
                raise ValueError('estimation_df has nan data!')
            else:

                series = pd.Series([None] * event_df.shape[0])
                series.index = event_df.index
                return series


if __name__ == '__main__':
    pass
