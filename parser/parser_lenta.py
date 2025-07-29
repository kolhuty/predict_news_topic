# Импорт библиотек
import requests as rq
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from IPython import display


class lentaRu_parser:
    def __init__(self):
        pass

    def _get_url(self, param_dict: dict) -> str:
        """
        Возвращает URL для запроса json таблицы со статьями

        url = 'https://lenta.ru/search/v2/process?'\
        + 'from=0&'\                       # Смещение
        + 'size=1000&'\                    # Кол-во статей
        + 'sort=2&'\                       # Сортировка по дате (2), по релевантности (1)
        + 'title_only=0&'\                 # Точная фраза в заголовке
        + 'domain=1&'\                     # ??
        + 'modified%2Cformat=yyyy-MM-dd&'\ # Формат даты
        + 'type=1&'\                       # Материалы. Все материалы (0). Новость (1)
        + 'bloc=4&'\                       # Рубрика. Экономика (4). Все рубрики (0)
        + 'modified%2Cfrom=2020-01-01&'\
        + 'modified%2Cto=2020-11-01&'\
        + 'query='                         # Поисковой запрос
        """
        hasType = int(param_dict['type']) != 0
        hasBloc = int(param_dict['bloc']) != 0

        url = 'https://lenta.ru/search/v2/process?' \
              + 'from={}&'.format(param_dict['from']) \
              + 'size={}&'.format(param_dict['size']) \
              + 'sort={}&'.format(param_dict['sort']) \
              + 'title_only={}&'.format(param_dict['title_only']) \
              + 'domain={}&'.format(param_dict['domain']) \
              + 'modified%2Cformat=yyyy-MM-dd&' \
              + 'type={}&'.format(param_dict['type']) * hasType \
              + 'bloc={}&'.format(param_dict['bloc']) * hasBloc \
              + 'modified%2Cfrom={}&'.format(param_dict['dateFrom']) \
              + 'modified%2Cto={}&'.format(param_dict['dateTo']) \
              + 'query={}'.format(param_dict['query'])

        return url

    def _get_search_table(self, param_dict: dict) -> pd.DataFrame:
        """
        Возвращает pd.DataFrame со списком статей
        """
        url = self._get_url(param_dict)
        r = rq.get(url)
        search_table = pd.DataFrame(r.json()['matches'])

        return search_table

    def get_articles(self,
                     param_dict,
                     time_step=10,
                     save_every=5,
                     save_excel=True) -> pd.DataFrame:
        """
        Функция для скачивания статей интервалами через каждые time_step дней
        Делает сохранение таблицы через каждые save_every * time_step дней

        param_dict: dict
        ### Параметры запроса
        ###### project - раздел поиска, например, rbcnews
        ###### category - категория поиска, например, TopRbcRu_economics
        ###### dateFrom - с даты
        ###### dateTo - по дату
        ###### offset - смещение поисковой выдачи
        ###### limit - лимит статей, максимум 100
        ###### query - поисковой запрос (ключевое слово), например, РБК

        """
        param_copy = param_dict.copy()
        time_step = timedelta(days=time_step)
        dateFrom = datetime.strptime(param_copy['dateFrom'], '%Y-%m-%d')
        dateTo = datetime.strptime(param_copy['dateTo'], '%Y-%m-%d')
        if dateFrom > dateTo:
            raise ValueError('dateFrom should be less than dateTo')

        out = pd.DataFrame()
        save_counter = 0

        while dateFrom <= dateTo:
            param_copy['dateTo'] = (dateFrom + time_step).strftime('%Y-%m-%d')
            if dateFrom + time_step > dateTo:
                param_copy['dateTo'] = dateTo.strftime('%Y-%m-%d')
            print('Parsing articles from ' \
                  + param_copy['dateFrom'] + ' to ' + param_copy['dateTo'])
            out = pd.concat([out, self._get_search_table(param_copy)], axis=0, ignore_index=True)
            dateFrom += time_step + timedelta(days=1)
            param_copy['dateFrom'] = dateFrom.strftime('%Y-%m-%d')
            save_counter += 1
            if save_counter == save_every:
                display.clear_output(wait=True)
                out.to_excel("/tmp/checkpoint_table.xlsx")
                print('Checkpoint saved!')
                save_counter = 0

        if save_excel:
            out.to_excel("lenta_{}_{}.xlsx".format(
                param_dict['dateFrom'],
                param_dict['dateTo']))
        print('Finish')

        return out

# Задаем тут параметры
use_parser = "LentaRu"

query = ''
offset = 0
size = 500
sort = "3"
title_only = "0"
domain = "1"
material = "0"
bloc = "0"
dateFrom = '2020-07-01'
dateTo = "2023-12-28"

if use_parser == "LentaRu":
    param_dict = {'query'     : query,
                  'from'      : str(offset),
                  'size'      : str(size),
                  'dateFrom'  : dateFrom,
                  'dateTo'    : dateTo,
                  'sort'      : sort,
                  'title_only': title_only,
                  'type'      : material,
                  'bloc'      : bloc,
                  'domain'    : domain}

print(use_parser, "- param_dict:", param_dict)

# Тоже будем собирать итеративно, правда можно ставить time_step побольше, т.к.
# больше лимит на запрос статей. И Работает быстрее :)
assert use_parser == "LentaRu"
parser = lentaRu_parser()
tbl = parser.get_articles(param_dict=param_dict,
                         time_step = 2,
                         save_every = 500,
                         save_excel = True)

#tbl.head()
#tbl['topic'] = 6

tbl = tbl[tbl.bloc.isin([1, 37, 3, 4, 5, 8, 48, 87])]

TagsMap = {1 : 0, 3 : 3, 4 : 1, 5 : 8, 8 : 4, 37 : 2, 48 : 7, 87 : 5}

tbl['topic'] = tbl['bloc'].map(TagsMap)

df = pd.DataFrame({
    'content': tbl['title'] + ' ' + tbl['text'],
    'topic': tbl['topic']
})

query = 'строительство'
offset = 0
size = 500
sort = "3"
title_only = "0"
domain = "1"
material = "0"
bloc = "0"
dateFrom = '2020-07-01'
dateTo = "2023-12-28"

if use_parser == "LentaRu":
    param_dict = {'query'     : query,
                  'from'      : str(offset),
                  'size'      : str(size),
                  'dateFrom'  : dateFrom,
                  'dateTo'    : dateTo,
                  'sort'      : sort,
                  'title_only': title_only,
                  'type'      : material,
                  'bloc'      : bloc,
                  'domain'    : domain}

print(use_parser, "- param_dict:", param_dict)

# Тоже будем собирать итеративно, правда можно ставить time_step побольше, т.к.
# больше лимит на запрос статей. И Работает быстрее :)
assert use_parser == "LentaRu"
parser = lentaRu_parser()
tbl_build = parser.get_articles(param_dict=param_dict,
                         time_step = 50,
                         save_every = 500,
                         save_excel = True)

df_build = pd.DataFrame({
    'content': tbl_build['title'] + ' ' + tbl_build['text'],
    'topic': 6
})

df_combined = pd.concat([df, df_build], ignore_index=True)
df_combined = df_combined.drop_duplicates(subset=['content'], keep='first')

print(f"Количество статей: {len(df_combined)}")
#print(df_combined.sample(10))
print(df_combined['topic'].value_counts()) # распределение тем

df_combined.to_csv('lenta_archive.csv', index=False) # сохраняем данные в файл