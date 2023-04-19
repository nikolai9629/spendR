import dash
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Output, Input

import figures

external_stylesheets = [{'rel': 'stylesheet'}]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'spendR'

app.layout = html.Div([
    html.Div(
        children=[
            dcc.Location(id='url', refresh=False),
            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.Div(['spendR'], className='logo'),
                            dcc.Graph(
                                id='main-diag',
                                config={'displaylogo': False},
                                className='graph'
                            ),
                            dcc.Graph(
                                id='balance-graph',
                                config={'displaylogo': False},
                                className='graph'
                            )
                        ],
                        className='upper-content'
                    ),
                    dcc.Graph(id='multi-graph', config={'displaylogo': False}, className='graph')
                ],
                className='content'
            )
        ],
        className='page'
    )
])


@app.callback(
    Output('main-diag', 'figure'),
    Input('url', 'pathname')
)
def update_diag_figure(_):
    figure = figures.getdiagfig()
    return figure


@app.callback(
    Output('multi-graph', 'figure'),
    Input('url', 'pathname')
)
def update_multi_figure(_):
    figure = figures.getmultifig()
    return figure


@app.callback(
    Output('balance-graph', 'figure'),
    Input('url', 'pathname')
)
def update_big_graph(_):
    figure = figures.getbalancefig()
    return figure


if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port='7000')


