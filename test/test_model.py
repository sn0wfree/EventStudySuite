# coding=utf-8
import unittest

import pandas as pd

from EventStudySuite.core.es_base import EventStudy
from EventStudySuite.core.es_model import MarketModel


class MyTestCase(unittest.TestCase):
    def test_try(self):
        stock_list = ['AMZN', 'AAPL', 'TSLA', 'GE', 'GILD', 'BA', 'NFLX', 'MS',
                      'LNVGY', 'BABA', 'LK', 'JOBS', 'CEO', 'TSM', 'JD', '^GSPC']
        firm_list = stock_list[:-1]
        # date = ['2020-03-13', '2020-03-13', '2020-03-13', '2020-03-13', '2020-03-13', '2020-03-13', '2020-03-13',
        #         '2020-03-13', '2020-01-23', '2020-01-23', '2020-01-23', '2020-01-23', '2020-01-23', '2020-01-23',
        #         '2020-01-23']
        # date_df = pd.DataFrame(date, index=firm_list, columns=['Date'])
        # date_df.index.name = 'CompanyName'
        #
        return_df = pd.read_csv('data.csv')
        # event_dict = date_df.to_dict()['Date']
        event_dict = {'AMZN': ['2020-03-13', '2020-01-23'], 'AAPL': '2020-03-13', 'TSLA': '2020-03-13',
                      'GE': '2020-03-13',
                      'GILD': '2020-03-13', 'BA': '2020-03-13', 'NFLX': '2020-03-13', 'MS': '2020-03-13',
                      'LNVGY': '2020-01-23', 'BABA': '2020-01-23', 'LK': '2020-01-23', 'JOBS': '2020-01-23',
                      'CEO': '2020-01-23', 'TSM': '2020-01-23', 'JD': '2020-01-23'}
        # event_dict['t'] = 1
        es = EventStudy(model=MarketModel, event_info=(250, 10, 5, 1))
        res = es(return_df, event_dict=event_dict, date='Date', factors=['Mkt_RF'], )
        # np.sqrt(ar.iloc[:,9-1].std()**2/17)*1.96
        # print(es.aar)

        print(res)

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
