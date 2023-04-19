import numpy as np
import pandas as pd
import datetime as dt

import plotly.graph_objs as go
from plotly.subplots import make_subplots

from scipy import stats

import json

TYPES = [
    'food', 'medicines', 'entertainment',
    'delivery', 'taxi', 'rent', 'credit'
]


def load_data():
    with open('config.json', encoding='utf-8') as file:
        config = json.load(file)

    url = f'https://docs.google.com/spreadsheets/d/{config["id"]}/gviz/tq?tqx=out:csv'
    df = pd.read_csv(url, error_bad_lines=False)

    prepared_df = prepare_data(df)

    return prepared_df


def prepare_data(df):

    with open('names.json', encoding='utf-8') as file:
        names = json.load(file)

    df = df.rename(columns=names['columns'])
    for old_key, new_key in names['types'].items():
        df.loc[(df.type == old_key), 'type'] = new_key

    df['ds'] = df['ds'].apply(lambda ds: '-'.join(ds.split(".")[::-1]))
    df = df[['ds', 'type', 'value']]

    prepared_df = pd.DataFrame()
    for (ds, type), single_df in df.groupby(['ds', 'type']):
        prepared_df = pd.concat([
            prepared_df, pd.DataFrame({'ds': ds, 'type': type, 'value': [single_df['value'].sum()]})
        ])

    ds_min = dt.date(*[int(ds_i) for ds_i in df['ds'].min().split('-')])
    current_ds = dt.datetime.now().date()

    full_ds_list = [
        '-'.join([ds_i for ds_i in str(ds_min + dt.timedelta(days=x))[:10].split(".")])
        for x in range((current_ds-ds_min).days+1)
    ]
    all_types = [t for _, t in names['types'].items()]

    fullsize_empty_df = pd.DataFrame(
        data={'value': 0},
        index=pd.MultiIndex.from_product([
            full_ds_list, all_types], names=['ds', 'type'])
    )
    prepared_df = fullsize_empty_df.add(
                    prepared_df.set_index(['ds', 'type']), fill_value=0).reset_index()

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

        figure.add_trace(
            go.Scatter(
                x=type_df['ds'], y=(-1)*type_df['value'].values,
                line=dict(color=color), marker=dict(size=7),
                mode='lines+markers', name=type,
                showlegend=False, legendgroup=type
            ),
            row=1+sub_index//3, col=1+sub_index%3
        )

    return figure


def getdiagfig():
    df = load_data()
    df = df.loc[df.is_fit]

    values = [(-1)*df.loc[df.type == type]['value'].sum() for type in TYPES]

    colors = [
        '#636efa', '#ef553b', '#00cc96',
        '#ab63fa', '#ffa15a', '#19d3f3', '#daf7a6'

    ]
    figure = go.Figure(
        data=go.Pie(labels=TYPES, values=values, marker=dict(colors=colors))
    )

    return figure


def make_forecast_area(df, fixation_point):
    fit_df = df.loc[df.is_fit]
    predict_df = df.loc[df.is_fit != True]

    ds_list = fit_df['ds'].unique()

    use_date = dt.date(*[int(ds_i) for ds_i in ds_list[0].split('-')])
    extended_ds_list = [
        '-'.join([ds_i for ds_i in str(use_date + dt.timedelta(days=x))[:10].split(".")])
        for x in range(len(ds_list) + 30)
    ]

    forecast_dict = {
        'lower': {'x': extended_ds_list, 'y': np.zeros(len(extended_ds_list))},
        'upper': {'x': extended_ds_list, 'y': np.zeros(len(extended_ds_list))}
    }

    lower_mean, upper_mean = 0, fit_df.loc[(fit_df.type == 'food')]['value'].mean()

    for type in TYPES:
        if type not in ['medicines', 'rent', 'credit']:
            lower_mean += fit_df.loc[fit_df.type == type]['value'].mean()

    for f_key, f_mean in zip(['lower', 'upper'], [lower_mean, upper_mean]):
        apxmt_val, shift_val = 0, 0
        for ds_index, ds in enumerate(extended_ds_list):
            for _, row in predict_df.loc[predict_df.ds == ds].iterrows():
                shift_val += row['value']

            forecast_dict[f_key]['y'][ds_index] = apxmt_val + shift_val
            apxmt_val += f_mean

        forecast_dict[f_key]['y'] += (
            fixation_point - forecast_dict[f_key]['y'][ds_list.shape[0]-1]
        )

        for index in np.arange(0, ds_list.shape[0]-1):
            forecast_dict[f_key]['y'][index] = np.nan

    return forecast_dict


def make_norm_forecast_area(df, fixation_point):
    fit_df = df.loc[df.is_fit]
    predict_df = df.loc[df.is_fit != True]

    fit_ds_list = fit_df['ds'].unique()

    fit_mean_values = np.zeros(fit_ds_list.shape[0])
    for ds_index, (_, ds_fit_df) in enumerate(fit_df.groupby('ds')):
        ds_fit_mean = 0
        for type in TYPES:
            if type not in ['medicines', 'rent', 'credit']:
                ds_fit_mean += ds_fit_df.loc[ds_fit_df.type == type]['value'].values[0]
        fit_mean_values[ds_index] = ds_fit_mean

    loc = fit_mean_values.mean()
    scale = np.std(fit_mean_values)

    last_date = dt.date(*[int(ds_i) for ds_i in fit_ds_list[-1].split('-')])
    predict_ds_list = [
        '-'.join([ds_i for ds_i in str(last_date + dt.timedelta(days=x))[:10].split(".")])
        for x in range(31)
    ]

    sample_num, ds_num = 100000, len(predict_ds_list)-1
    forecast_samples = stats.norm.rvs(loc=loc, scale=scale, size=sample_num*ds_num).reshape(sample_num, ds_num)

    fit_min, fit_max = fit_mean_values.min(), fit_mean_values[fit_mean_values != 0].max()
    ####
    fit_max = -5000

    forecast_samples[forecast_samples > fit_max] = fit_max
    forecast_samples[forecast_samples < fit_min] = fit_min

    upper_sample_value, lower_sample_value = forecast_samples[0].sum(), forecast_samples[0].sum()
    upperr_sample_index, lower_sample_index = 0, 0

    for sample_index in range(1, forecast_samples.shape[0]):
        pot_extremum_value = forecast_samples[sample_index].sum()

        if pot_extremum_value > upper_sample_value:
            upper_sample_value = pot_extremum_value
            upperr_sample_index = sample_index

        if pot_extremum_value < lower_sample_value:
            lower_sample_value = pot_extremum_value
            lower_sample_index = sample_index

    forecast_dict = {
        'lower': {'x': predict_ds_list, 'y': np.zeros(len(predict_ds_list))},
        'upper': {'x': predict_ds_list, 'y': np.zeros(len(predict_ds_list))}
    }

    for f_key, sample_index in zip(['lower', 'upper'], [lower_sample_index, upperr_sample_index]):
        apxmt_val, shift_val = fixation_point, 0
        forecast_dict[f_key]['y'][0] = apxmt_val
        for ds_index, lower_value in enumerate(forecast_samples[sample_index]):
            for _, row in predict_df.loc[predict_df.ds == predict_ds_list[ds_index+1]].iterrows():
                shift_val += row['value']

            if apxmt_val + shift_val + lower_value < 0:
                apxmt_val -= apxmt_val + shift_val
            else:
                apxmt_val += lower_value

            forecast_dict[f_key]['y'][ds_index+1] = apxmt_val + shift_val

    return forecast_dict


def make_balance_line(df):
    fit_df = df.loc[df.is_fit]

    ds_list, balance_values = [], []

    balance = 0
    for ds, ds_data in fit_df.groupby('ds'):
        ds_list.append(ds)
        for _, row in ds_data.iterrows():
            balance += row['value']
        balance_values.append(balance)

    return {'x': ds_list, 'y': balance_values}


def getbalancefig():
    df = load_data()

    balance_dict = make_balance_line(df)
    forecast_dict = make_norm_forecast_area(df, balance_dict['y'][-1])


    figure = make_subplots(cols=1, rows=1, subplot_titles=['balance'])

    figure.add_trace(
        go.Scatter(
            x=forecast_dict['upper']['x'], y=forecast_dict['upper']['y'],
            line=dict(color='rgba(99, 110, 252, 0.35)', width=1),
            marker=dict(size=5), name='food_aprxmtn', mode='lines',
            showlegend=False, legendgroup='food_aprxmtn'
        ), row=1, col=1
    )

    figure.add_trace(
        go.Scatter(
            x=forecast_dict['lower']['x'], y=forecast_dict['lower']['y'],
            line=dict(color='rgba(99, 110, 252, 0.35)', width=1),
            fillcolor='rgba(99, 110, 252, 0.25)', fill='tonexty',
            marker=dict(size=5), name='full_aprxmtn', mode='lines',
            showlegend=False, legendgroup='lower'
        ), row=1, col=1
    )

    figure.add_trace(
        go.Scatter(
            x=balance_dict['x'], y=balance_dict['y'],
            line=dict(color='#1f4770'), marker=dict(size=7), name='balance',
            mode='lines+markers', showlegend=False, legendgroup='upper'
        ), row=1, col=1
    )

    return figure
