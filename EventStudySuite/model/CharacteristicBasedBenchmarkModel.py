# coding=utf-8
import numpy as np
from EventStudySuite.model.DefaultModel import DefaultModel


class CharacteristicBasedBenchmarkModel(DefaultModel):
    @staticmethod
    def cal_ar(estimation_df, event_df, stock: str, formula: str = "w:0.4,0.5,0.6", factors=['Mkt'],
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
        if formula.startswith('w'):
            weight = formula[2:].split(',')
        else:
            raise ValueError('formula for CharacteristicBasedBenchmarkModel is wrong! should start with w:')

        len_weight = len(weight)
        if len_weight != len(factors):
            if len_weight == 1 or 0:
                weight = [1 / len(factors)] * len(factors)
            else:
                raise ValueError('weight is not consistent with factors')
        else:
            weight = list(map(float, weight))
        twmp = np.dot(event_df[factors], weight)

        # avg = estimation_df[stock].mean()
        #
        # return event_df[stock] - avg
        return event_df[stock] - twmp


if __name__ == '__main__':
    formula: str = "w:0.4,0.5,0.6"
    if formula.startswith('w'):
        weight = formula[2:].split(',')
    else:
        raise ValueError('formula for CharacteristicBasedBenchmarkModel is wrong! should start with w:')
    print(weight)
    pass
