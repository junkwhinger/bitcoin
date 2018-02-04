from bs4 import BeautifulSoup
import requests
import pandas as pd
import logging
import re
import datetime

class currency:

    def __init__(self, name):
        self.name = name
        self.__crawl_data()

    def crawl_data(self):
        from_date = '20130101'
        to_date = datetime.datetime.now().strftime("%Y%m%d")

        logging.info('crawling {} from {} to {}'.format(self.name, from_date, to_date))

        path = 'https://coinmarketcap.com/currencies/{}/historical-data/?start={}&end={}'.format(self.name,from_date, to_date)
        r = requests.get(path)
        data = r.text

        soup = BeautifulSoup(data, "html5lib")
        raw = soup.find_all("tr", "text-right")

        row_list = []
        for r in raw:
            tmp = {}
            for idx, n in enumerate(r):
                try:
                    tmp[idx] = n.text
                except:
                    pass
            row_list.append(tmp)

        df = pd.DataFrame(row_list)
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap']
        df.Date = df.Date.map(pd.to_datetime).dt.date

        df.Open = df.Open.map(pd.to_numeric)
        df.High = df.High.map(pd.to_numeric)
        df.Low = df.Low.map(pd.to_numeric)
        df.Close = df.Close.map(pd.to_numeric)

        df = df.sort_values(by='Date')
        df.set_index('Date', inplace=True)
        self.data = df

    __crawl_data = crawl_data


def crawl_currency_names(num_of_pages):

    logging.debug('crawling currency names on {} pages'.format(num_of_pages))

    currencies = []

    for i in range(num_of_pages):
        path = 'https://coinmarketcap.com/coins/{}'.format(i+1)
        r = requests.get(path)
        data = r.text

        soup = BeautifulSoup(data, "html5lib")
        raw = soup.find_all("a", "currency-name-container")

        for r in raw:
            href_text = r['href'].strip()
            m = re.search('/currencies/(.+?)/', href_text).group(1)
            currencies.append(m)

    return currencies