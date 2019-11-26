from plotly import tools
import plotly.io as pio
import numpy as np
from statistics import mean
from plotly.graph_objs import *

pio.orca.config.executable = 'C:\\Users\\Arunavo Ray\\AppData\\Local\\Programs\\orca\\orca.exe'


def plot_train_test_reward(train_rewards, test_rewards, file_name):

    figure = tools.make_subplots(rows=1, cols=2, subplot_titles=('Train Daily Avg Profit Per', 'Test Daily Avg Profit Per'), print_grid=False)
    figure.append_trace(Scatter(y=train_rewards, mode='lines', line=dict(color='skyblue')), 1, 1)
    figure.append_trace(Scatter(y=test_rewards, mode='lines', line=dict(color='orange')), 1, 2)
    # figure['layout']['xaxis1'].update(title='epoch')
    # figure['layout']['xaxis2'].update(title='epoch')
    figure['layout'].update(height=400, width=900, showlegend=False)
    pio.write_image(figure, file=file_name, format='png')


def plot_train_test_actions(genome_id, train_env, test_env, train_acts, test_acts, date_split, file_name):

    # plot
    train_copy = train_env.data.copy()
    test_copy = test_env.data.copy()
    train_copy['act'] = train_acts + [np.nan]
    test_copy['act'] = test_acts + [np.nan]
    train0 = train_copy[train_copy['act'] == 0]
    train1 = train_copy[train_copy['act'] == 1]
    train2 = train_copy[train_copy['act'] == 2]
    test0 = test_copy[test_copy['act'] == 0]
    test1 = test_copy[test_copy['act'] == 1]
    test2 = test_copy[test_copy['act'] == 2]
    act_color0, act_color1, act_color2 = 'gray', 'green', 'red'

    data = [
        Candlestick(x=train0.index, open=train0['Open'], high=train0['High'], low=train0['Low'], close=train0['Close'],
                    increasing=dict(line=dict(color=act_color0)), decreasing=dict(line=dict(color=act_color0))),
        Candlestick(x=train1.index, open=train1['Open'], high=train1['High'], low=train1['Low'], close=train1['Close'],
                    increasing=dict(line=dict(color=act_color1)), decreasing=dict(line=dict(color=act_color1))),
        Candlestick(x=train2.index, open=train2['Open'], high=train2['High'], low=train2['Low'], close=train2['Close'],
                    increasing=dict(line=dict(color=act_color2)), decreasing=dict(line=dict(color=act_color2))),
        Candlestick(x=test0.index, open=test0['Open'], high=test0['High'], low=test0['Low'], close=test0['Close'],
                    increasing=dict(line=dict(color=act_color0)), decreasing=dict(line=dict(color=act_color0))),
        Candlestick(x=test1.index, open=test1['Open'], high=test1['High'], low=test1['Low'], close=test1['Close'],
                    increasing=dict(line=dict(color=act_color1)), decreasing=dict(line=dict(color=act_color1))),
        Candlestick(x=test2.index, open=test2['Open'], high=test2['High'], low=test2['Low'], close=test2['Close'],
                    increasing=dict(line=dict(color=act_color2)), decreasing=dict(line=dict(color=act_color2)))
    ]
    title = 'Genome: {}| Train Profits : {} ( {}%), W : {} L : {} || Test Profits : {} ( {}%), W : {} L : {} '.format(
        genome_id,
        round(train_env.amt, 2),
        round(mean(train_env.daily_profit_per), 3),
        int(train_env.wins),
        int(train_env.losses),
        round(test_env.amt, 2),
        round(mean(test_env.daily_profit_per), 3),
        int(test_env.wins),
        int(test_env.losses)
    )
    layout = {
        'title':title,
        'showlegend': False,
        'shapes': [
            {'x0': date_split, 'x1': date_split, 'y0': 0, 'y1': 1, 'xref': 'x', 'yref': 'paper',
             'line': {'color': 'rgb(0,0,0)', 'width': 1}}
        ],
        'annotations': [
            {'x': date_split, 'y': 1.0, 'xref': 'x', 'yref': 'paper', 'showarrow': False, 'xanchor': 'left',
             'text': ' test data'},
            {'x': date_split, 'y': 1.0, 'xref': 'x', 'yref': 'paper', 'showarrow': False, 'xanchor': 'right',
             'text': 'train data '}
        ]
    }
    figure = Figure(data=data, layout=layout)
    pio.write_image(figure, file=file_name, width=1400, height=900 , scale=2,format='png')

    return title
