# coding=utf-8

from EventStudySuite.model.DefaultModel import DefaultModel


class ConstantMeanReturnModel(DefaultModel):
    @staticmethod
    def cal_ar(estimation_df, event_df, stock: str, formula: str = "{stock} - {factors}", factors=['Mkt'],
               raise_error=False):
        """
        均值常数模型（Constant Mean Return model，简写 CMRM），该方法假设个股的收益率总是围绕一个常数值做上下波动
        :param raise_error:
        :param estimation_df:  pd.DataFrame
        :param event_df:  pd.DataFrame
        :param stock:
        :param formula:
        :param factors:
        :return:
        """
        avg = estimation_df[stock].mean()

        return event_df[stock] - avg
