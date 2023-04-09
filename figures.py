import numpy as np
import pandas as pd
import datetime as dt

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import json

TYPES = ['food', 'medicines', 'entertainment', 'delivery', 'taxi', 'rent']

def load_data():
    url = 'https://docs.google.com/spreadsheets/d/1lqL5rmsCBaVCv43UxF1YQpxRzp9b8WfbgF2ufKlcnj8/gviz/tq?tqx=out:csv'
    df = pd.read_csv(url, error_bad_lines=False)

    with open('names.json', encoding='utf-8') as file:
        names = json.load(file)

    df = df.rename(columns=names['columns'])
    for old_key, new_key in names['types'].items():
        df.loc[(df.type == old_key), 'type'] = new_key

    df['ds'] = df['ds'].apply(lambda ds: '-'.join([ds_i for ds_i in ds.split(".")[::-1]]))
    df = df[['ds', 'type', 'value']]

    current_ds = str(dt.datetime.now())[:10]
    prepared_df = pd.DataFrame()
    for (ds, type), single_df in df.groupby(['ds', 'type']):
        if ds > current_ds:
            break
        prepared_df = pd.concat([
            prepared_df, pd.DataFrame({'ds': ds, 'type': type, 'value': [single_df['value'].sum()]})
        ])

    ds_min = dt.date(*[int(ds_i) for ds_i in df['ds'].min().split('-')])
    current_ds = dt.date(*[int(ds_i) for ds_i in current_ds.split('-')])
    # ds_max = dt.date(*[int(ds_i) for ds_i in df['ds'].max().split('-')])

    full_ds_list = ['-'.join([ds_i for ds_i in str(ds_min + dt.timedelta(days=x))[:10].split(".")]) for x in range((current_ds-ds_min).days+1)]
    all_types = [t for _, t in names['types'].items()]

    fullsize_empty_df = pd.DataFrame(
        data={'value': 0}, index=pd.MultiIndex.from_product([full_ds_list, all_types], names=['ds', 'type'])
    )
    prepared_df = fullsize_empty_df.add(prepared_df.set_index(['ds', 'type']), fill_value=0).reset_index()

    return prepared_df


def getmultifig():
    df = load_data()

    colors = ['#636efa', '#ef553b', '#00cc96', '#ab63fa', '#ffa15a', '#19d3f3']
    figure = make_subplots(rows=2, cols=3, subplot_titles=TYPES)

    for sub_index, (type, color) in enumerate(zip(TYPES, colors)):
        type_df = df.loc[df.type == type]
        # type_df.loc[(type_df.value == 0), 'value'] = np.nan

        figure.add_trace(
            go.Scatter(x=type_df['ds'], y=(-1)*type_df['value'].values, line=dict(color=color), marker=dict(size=7), name=type, mode='lines+markers', showlegend=False, legendgroup=type),
            row=1+sub_index//3, col=1+sub_index%3
        )

    return figure


def getdiagfig():
    df = load_data()

    values = []
    for type in TYPES:
        values.append((-1)*df.loc[df.type == type]['value'].sum())

    figure = go.Figure(data=go.Pie(
        labels=TYPES,
        values=values
    ))
    return figure


def getbalancefig():
    df = load_data()
    figure = make_subplots(cols=1, rows=1, subplot_titles=['balance'])

    ds_list, balance_values = [], []

    balance = 0
    for ds, ds_data in df.groupby('ds'):
        ds_list.append(ds)
        for _, row in ds_data.iterrows():
            balance += row['value']
        balance_values.append(balance)

    use_date = dt.date(*[int(ds_i) for ds_i in ds_list[0].split('-')])
    extended_ds_list = ['-'.join([ds_i for ds_i in str(use_date + dt.timedelta(days=x))[:10].split(".")]) for x in range(len(ds_list) + 7)]

    food_mean = df.loc[df.type == 'food']['value'].mean()

    full_mean = 0
    for type in TYPES:
        full_mean += df.loc[df.type == type]['value'].mean()

    food_apxmt_line, food_apxmt_val = np.zeros(len(extended_ds_list)), 0
    full_apxmt_line, full_apxmt_val = np.zeros(len(extended_ds_list)), 0

    for ds_index, _ in enumerate(extended_ds_list):
        food_apxmt_line[ds_index] = food_apxmt_val
        food_apxmt_val += food_mean

        full_apxmt_line[ds_index] = full_apxmt_val
        full_apxmt_val += full_mean

    food_apxmt_line += balance_values[len(balance_values)-1] - food_apxmt_line[len(balance_values)-1]
    full_apxmt_line += balance_values[len(balance_values)-1] - full_apxmt_line[len(balance_values)-1]

    for index in np.arange(0, len(balance_values)-1):
        food_apxmt_line[index] = np.nan
        full_apxmt_line[index] = np.nan

    figure.add_trace(
        go.Scatter(
            x=extended_ds_list, y=food_apxmt_line,
            line=dict(color='rgba(99, 110, 252, 0.25)', width=1),
            marker=dict(size=5), name='food_aprxmtn', mode='lines', showlegend=False, legendgroup='food_aprxmtn'
        ), row=1, col=1
    )

    figure.add_trace(
        go.Scatter(
            x=extended_ds_list, y=full_apxmt_line, fillcolor='rgba(99, 110, 252, 0.25)', fill='tonexty',
            line=dict(color='rgba(99, 110, 252, 0.25)', width=1),
            marker=dict(size=5), name='full_aprxmtn', mode='lines', showlegend=False, legendgroup='full_aprxmtn'
        ), row=1, col=1
    )

    figure.add_trace(
        go.Scatter(
            x=ds_list, y=balance_values, line=dict(color='#1f4770'),
            marker=dict(size=7), name='balance', mode='lines+markers', showlegend=False, legendgroup='balance'
        ), row=1, col=1
    )

    return figure

# blue #636efa
# red #ef553b
# green #00cc96
# purple #ab63fa
# orange #ffa15a
# cyan #19d3f3