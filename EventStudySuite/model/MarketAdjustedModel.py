# coding=utf-8

from EventStudySuite.model.DefaultModel import DefaultModel


class MarketAdjustedModel(DefaultModel):
    @staticmethod
    def cal_ar(estimation_df, event_df, stock: str, formula: str = "{stock} - {factors}", factors=['Mkt'],
               raise_error=False):
        """
市场调整法（Market Adjusted Model, 简写 MAM），也就是国内实务研究中常用的超额收益法，选定一个基准指数（全市场的宽基指数、风格指数或行业指数）
        :param raise_error:
        :param estimation_df:  pd.DataFrame
        :param event_df:  pd.DataFrame
        :param stock:
        :param formula:
        :param factors:
        :return:
        """
        return event_df.eval(formula.format(stock=stock, factors="+".join(factors)))
        # f1 = formula.format(stock=stock, factors="+".join(factors))
        # y_array = estimation_df[stock]
        # w = 1 - y_array.isna() * 1
        # length = estimation_df.shape[0]
        # if w.sum() == length:
        #     y_array = y_array.values.ravel()
        #     x_array = estimation_df[factors[0]].values.ravel()
        #     sum_y = np.sum(y_array)
        #     sum_x = np.sum(x_array)
        #     sum_xy = np.do(x_array, y_array.T)
        #     sum_x2 = np.dot(x_array, x_array)
        #     beta_Mkt = ((sum_x * sum_y) / length - sum_xy) / ((sum_x * sum_x) / length - sum_x2)
        #     alpha = (sum_y - sum_x * beta_Mkt) / length
        #     event_df[f'{stock}_er'] = np.dot(event_df[factors].values, beta_Mkt) + alpha
        #     # event_df[f'{stock}_ar'] = event_df.eval(f'{stock} - {stock}_er')
        #     series = event_df.eval(f'{stock} - {stock}_er')
        #     series.index = event_df.index
        #     return series
        # else:
        #     series = pd.Series([None] * event_df.shape[0])
        #     series.index = event_df.index
        #     return series


if __name__ == '__main__':
    pass
