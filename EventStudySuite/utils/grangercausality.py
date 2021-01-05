# coding=utf-8

from collections import namedtuple
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

def gc(df, maxlag, addconst=True, verbose=False, thershold=0.05):
    ssr_ftest_explain = ['teststatistic', 'pvalue', 'u', 'degrees_of_freedom']
    s = namedtuple('ssr_ftest', ssr_ftest_explain)
    res = grangercausalitytests(df, maxlag, addconst=addconst, verbose=verbose)
    x1, x2 = df.columns
    h = []
    for lag, v_res in res.items():
        ft = s(*v_res[0]['ssr_ftest'])
        # print(lag, v_res)
        if ft.pvalue > thershold:
            status = False
        else:
            status = True
        h.append([x1, x2, lag, status])
    return pd.DataFrame(h, columns=['var1', 'var2', 'lag', 'var1 does Granger cause var2'])


if __name__ == '__main__':
    # ssr_ftest_explain = ['teststatistic', 'pvalue', 'u', 'degrees_of_freedom']
    # s = namedtuple('ssr_ftest', ssr_ftest_explain)
    # thershold = 0.05
    df = pd.DataFrame([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1.21, 2.321, 3.41, 8, 5, 6, 7, 0, 9, 10]],
                      index=['x1', 'x2']).T
    s = gc(df, 2)
    print(s)
    # res = gc(df, 2)
    # x1, x2 = df.columns
    # for lag, v_res in res.items():
    #     ft = s(*v_res[0]['ssr_ftest'])
    #     print(lag, v_res)
    #     if ft.pvalue > thershold:
    #         status = False
    #     else:
    #         status = True
    #
    #     x1, x2, lag, status
    # print(res)
    pass
