import numpy as np
import pandas as pd
import datetime as dt

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import json

TYPES = ['food', 'medicines', 'entertainment', 'delivery', 'taxi', 'rent', 'credit']

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

    prepared_df = pd.DataFrame()
    for (ds, type), single_df in df.groupby(['ds', 'type']):
        prepared_df = pd.concat([
            prepared_df, pd.DataFrame({'ds': ds, 'type': type, 'value': [single_df['value'].sum()]})
        ])

    ds_min = dt.date(*[int(ds_i) for ds_i in df['ds'].min().split('-')])
    ds_max = dt.date(*[int(ds_i) for ds_i in df['ds'].max().split('-')])

    current_ds = dt.datetime.now().date()

    full_ds_list = ['-'.join([ds_i for ds_i in str(ds_min + dt.timedelta(days=x))[:10].split(".")]) for x in range((current_ds-ds_min).days+1)]
    all_types = [t for _, t in names['types'].items()]

    fullsize_empty_df = pd.DataFrame(
        data={'value': 0}, index=pd.MultiIndex.from_product([full_ds_list, all_types], names=['ds', 'type'])
    )
    prepared_df = fullsize_empty_df.add(prepared_df.set_index(['ds', 'type']), fill_value=0).reset_index()

    prepared_df.loc[(prepared_df.ds <= str(current_ds)), 'is_fit'] = True
    prepared_df.loc[(prepared_df.ds > str(current_ds)), 'is_fit'] = False

    return prepared_df


def getmultifig():
    df = load_data()

    df = df.loc[df.is_fit]
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
    df = df.loc[df.is_fit]

    values = []
    for type in TYPES:
        values.append((-1)*df.loc[df.type == type]['value'].sum())

    colors = ['#636efa', '#ef553b', '#00cc96', '#ab63fa', '#ffa15a', '#19d3f3', '#DAF7A6']
    figure = go.Figure(data=go.Pie(labels=TYPES, values=values, marker=dict(colors=colors)))

    return figure


def getbalancefig():
    df = load_data()
    fit_df = df.loc[df.is_fit]

    figure = make_subplots(cols=1, rows=1, subplot_titles=['balance'])

    ds_list, balance_values = [], []

    balance = 0
    for ds, ds_data in fit_df.groupby('ds'):
        ds_list.append(ds)
        for _, row in ds_data.iterrows():
            balance += row['value']
        balance_values.append(balance)

    use_date = dt.date(*[int(ds_i) for ds_i in ds_list[0].split('-')])
    extended_ds_list = ['-'.join([ds_i for ds_i in str(use_date + dt.timedelta(days=x))[:10].split(".")]) for x in range(len(ds_list) + 30)]

    ##### forecast
    predict_df = df.loc[df.is_fit != True]

    forecast_dict = {'lower': np.zeros(len(extended_ds_list)), 'upper': np.zeros(len(extended_ds_list))}
    lower_mean, upper_mean = 0, fit_df.loc[(fit_df.type == 'food')]['value'].mean()

    for type in TYPES:
        if type not in ['rent', 'credit']:
            lower_mean += fit_df.loc[fit_df.type == type]['value'].mean()

    for f_key, f_mean in zip(forecast_dict, [lower_mean, upper_mean]):
        apxmt_val, shift_val = 0, 0
        for ds_index, ds in enumerate(extended_ds_list):
            for _, row in predict_df.loc[predict_df.ds == ds].iterrows():
                shift_val += row['value']

            forecast_dict[f_key][ds_index] = apxmt_val + shift_val
            apxmt_val += f_mean

        forecast_dict[f_key] += balance_values[len(balance_values)-1] - forecast_dict[f_key][len(balance_values)-1]

        for index in np.arange(0, len(balance_values)-1):
            forecast_dict[f_key][index] = np.nan

    figure.add_trace(
        go.Scatter(
            x=extended_ds_list, y=forecast_dict['upper'],
            line=dict(color='rgba(99, 110, 252, 0.25)', width=1),
            marker=dict(size=5), name='food_aprxmtn', mode='lines', showlegend=False, legendgroup='food_aprxmtn'
        ), row=1, col=1
    )

    figure.add_trace(
        go.Scatter(
            x=extended_ds_list, y=forecast_dict['lower'], fillcolor='rgba(99, 110, 252, 0.25)', fill='tonexty',
            line=dict(color='rgba(99, 110, 252, 0.25)', width=1),
            marker=dict(size=5), name='full_aprxmtn', mode='lines', showlegend=False, legendgroup='lower'
        ), row=1, col=1
    )

    figure.add_trace(
        go.Scatter(
            x=ds_list, y=balance_values, line=dict(color='#1f4770'),
            marker=dict(size=7), name='balance', mode='lines+markers', showlegend=False, legendgroup='upper'
        ), row=1, col=1
    )

    return figure

# blue #636efa
# red #ef553b
# green #00cc96
# purple #ab63fa
# orange #ffa15a
# cyan #19d3f3