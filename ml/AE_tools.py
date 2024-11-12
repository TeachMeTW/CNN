import pandas as pd
import numpy as np
import tensorflow as tf
import logging
import warnings
import plotly.express as px
import plotly.graph_objs as go
from sklearn import preprocessing
from sklearn.preprocessing import FunctionTransformer
from sklearn import metrics
from sklearn.manifold import TSNE

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)
warnings.filterwarnings('ignore')

color_scale = ['green', 'lightgray', 'orange']

def prep_data(data_frame, features, scaler):
    data_frame[features] = data_frame[features].astype(float)
    data_frame[features] = data_frame[features].fillna(0.0)
    if scaler == 'MinMax':
        scaler_ = preprocessing.MinMaxScaler(feature_range=(0, 1))
    elif scaler == 'Robust':
        scaler_ = preprocessing.RobustScaler()
    elif scaler == 'Standard':
        scaler_ = preprocessing.StandardScaler()
    elif scaler == 'Log10':
        scaler_ = FunctionTransformer(np.log10, check_inverse=True, validate=True)
    else:
        scaler_ = None
    data_frame[features] = scaler_.fit_transform(data_frame[features])
    data_frame[features] = data_frame[features].fillna(0.0)
    data_frame[features] = data_frame[features].replace([np.inf, -np.inf], 0.0)
    return data_frame, scaler_


def build_latent_space(auto_encoder, data_frame, features):
    latent_layer = int(len(auto_encoder.layers) / 2) + 1
    encoder = tf.keras.models.Sequential(auto_encoder.layers[:latent_layer])
    latent_space = encoder.predict(data_frame[features].values)
    latent_space = pd.DataFrame(latent_space)

    latent_space_names = []
    i = 1
    for dim in range(0, latent_space.shape[1]):
        dim_name = 'Latent_' + str(i)
        latent_space_names.append(dim_name)
        i += 1

    latent_space.columns = latent_space_names
    return latent_space


def plot_latent_space(data_frame, color, hover_data, title, template, height=None, width=None, auto_size=True):
    lds = data_frame.filter(regex='Latent_').columns.to_list()
    if len(lds) > 2:
        fig = px.scatter_3d(data_frame,
                            x='Latent_1',
                            y='Latent_2',
                            z='Latent_3',
                            color=color,
                            hover_data=hover_data,
                            title=title,
                            color_continuous_scale=['blue', 'lightgray', 'red'],
                            color_discrete_map={'GREEN': 'green',
                                                'YELLOW': 'gold',
                                                'RED': 'red'
                                                },
                            color_discrete_sequence=px.colors.qualitative.G10)
        fig.update_layout(autosize=auto_size,
                          height=height,
                          width=width,
                          template=template)
        fig.update_xaxes(tickangle=45)
        #fig.show()
    else:
        fig = px.scatter(data_frame,
                         x='Latent_1',
                         y='Latent_2',
                         color=color,
                         hover_data=hover_data,
                         title=title,
                         color_continuous_scale=['blue', 'lightgray', 'red'],
                         color_discrete_map={'GREEN': 'green',
                                             'YELLOW': 'gold',
                                             'RED': 'red'
                                             },
                         color_discrete_sequence=px.colors.qualitative.G10)
        fig.update_layout(autosize=auto_size,
                          height=height,
                          width=width,
                          template=template)
        fig.update_xaxes(tickangle=45)
        #fig.show()
    return fig


def calculate_error_threshold(auto_encoder, train_data, features, percentile):
    post_hoc = auto_encoder.predict(train_data[features])
    train_data['reconstruction_error'] = np.mean(np.power(train_data[features].values - post_hoc, 2), axis=1)
    train_data['reconstruction_error'] = np.log10(train_data['reconstruction_error']).fillna(0.0)
    error_threshold = np.percentile(train_data['reconstruction_error'], percentile)
    return train_data, error_threshold


def calculate_error(auto_encoder, data_frame, features):
    X = data_frame[features]
    preds = auto_encoder.predict(X)
    data_frame['reconstruction_error'] = np.mean(np.power(X - preds, 2), axis=1)
    data_frame['reconstruction_error'] = np.log10(data_frame['reconstruction_error']).fillna(0.0)
    return data_frame, pd.DataFrame(preds, columns=features)


def plot_loss(history, title, template, height=None, width=None, auto_size=True):
    print('N overfit: {}'.format(history[history['val_loss'] > history['loss']].shape))
    fig = go.Figure()
    loss = ['loss', 'val_loss']
    for l in loss:
        color = 'red'
        if l == 'val_loss':
            color = 'blue'
        else:
            pass
        fig.add_trace(go.Scatter(x=history.index,
                                 y=history[l],
                                 mode='lines+markers',
                                 line_color=color,
                                 name=l))
    fig.update_layout(autosize=auto_size,
                      height=height,
                      width=width,
                      template=template,
                      title={'text': title},
                      xaxis_title="Epoch",
                      yaxis_title="Loss")
    fig.update_xaxes(tickangle=45)
    #fig.show()
    return fig


def plot_error_over_time(df, timestamp_col, id_col, title, color_overlay, error_threshold, template, height=None, width=None,
                         auto_size=True):
    HOVER_DATA = [timestamp_col, id_col, color_overlay]
    fig = px.scatter(df,
                     x=timestamp_col,
                     y='reconstruction_error',
                     color=color_overlay,
                     hover_data=HOVER_DATA,
                     color_continuous_scale=['blue', 'lightgray', 'red'],
                     color_discrete_sequence=px.colors.qualitative.G10,
                     title=title
                     )
    fig.add_hline(y=error_threshold,
                  line_width=1,
                  line_dash='dash',
                  line_color='red',
                  name='Anomaly Threshold',
                  opacity=1.0)
    fig.update_layout(autosize=auto_size,
                      height=height,
                      width=width,
                      template=template)
    fig.update_xaxes(tickangle=45)
    fig.update_traces(opacity=1.0)
    print(f'Reconstruction Error Threshold: {error_threshold}')
    #fig.show()
    return fig


def compute_error_per_dim(point, input_dim, df, reconstructions):
    df_np = df.values
    reconstructions_np = reconstructions.values
    initial_pt = df_np[point].reshape(1, input_dim)
    reconstructed_pt = reconstructions_np[point].reshape(1, input_dim)
    return abs((initial_pt - reconstructed_pt)[0])


def plot_mean_error(error_frame, template, title, height=None, width=None, auto_size=True):
    feature_errors = error_frame.describe().T.sort_values('mean', ascending=False).reset_index()
    feature_errors.rename(columns={'index': 'sensor', 'mean': 'avg_reconstruction_error'}, inplace=True)
    feature_errors = feature_errors[['sensor', 'avg_reconstruction_error']].sort_values('avg_reconstruction_error',
                                                                                      ascending=False)

    fig = px.bar(feature_errors,
                 x='sensor',
                 y='avg_reconstruction_error',
                 text_auto='.2s',
                 title=title,
                 color='avg_reconstruction_error',
                 color_continuous_scale=['blue', 'lightgray', 'red'])
    # fig.add_hline(y=error_threshold, # error_threshold
    #              line_width=1,
    #              line_dash='dash',
    #              line_color='red',
    #              name='Anomaly Threshold',
    #              opacity=1.0)
    fig.update_layout(autosize=auto_size,
                      height=height,
                      width=width,
                      template=template)
    fig.update_xaxes(tickangle=45)
    fig.update_traces(opacity=1.0,
                      textfont_size=12,
                      textangle=0,
                      textposition="outside",
                      cliponaxis=False)
    #fig.show()
    return fig


def plot_error_per_feature(error_record, title, template, height=None, width=None, auto_size=True):
    error_record = error_record.sort_values('reconstruction_error', ascending=False)
    fig = px.bar(error_record,
                 x='sensor',
                 y='reconstruction_error',
                 text_auto='.2s',
                 color='reconstruction_error',
                 color_continuous_scale=['blue', 'lightgray', 'red'],
                 title=title)
    fig.update_layout(autosize=auto_size,
                      height=height,
                      width=width,
                      template=template)
    fig.update_traces(opacity=1.0,
                      textfont_size=12,
                      textangle=0,
                      textposition="outside",
                      cliponaxis=False)
    fig.update_xaxes(tickangle=45)
    #fig.show()
    return fig


def build_tsne(best_params, data_frame, features, perplexity):
    if best_params['latent_dim'] > 3:
        n_components = 3
    else:
        n_components = best_params['latent_dim']
    t = TSNE(perplexity=perplexity,
             n_components=n_components,
             verbose=True,
             random_state=7).fit_transform(data_frame[features])
    t = pd.DataFrame(t)
    tSNE_col_names = []
    i = 1
    for dim in range(0, t.shape[1]):
        dim_name = 'tSNE_' + str(i)
        tSNE_col_names.append(dim_name)
        i += 1
    t.columns = tSNE_col_names
    return t


def plot_tsne(df, hover_data, color, title, template, height=None, width=None, auto_size=True):
    param_len = df.filter(regex='tSNE_').shape[1]

    if param_len == 2:
        fig = px.scatter(df,
                         x='tSNE_1',
                         y='tSNE_2',
                         color=color,
                         hover_data=hover_data,
                         title=title,
                         color_continuous_scale=['blue', 'lightgray', 'red'],
                         color_discrete_map={'GREEN': 'green',
                                             'YELLOW': 'gold',
                                             'RED': 'red'
                                             },
                         color_discrete_sequence=px.colors.qualitative.G10)
    else:
        fig = px.scatter_3d(df,
                            x='tSNE_1',
                            y='tSNE_2',
                            z='tSNE_3',
                            color=color,
                            hover_data=hover_data,
                            title=title,
                            color_continuous_scale=['blue', 'lightgray', 'red'],
                            color_discrete_map={'GREEN': 'green',
                                                'YELLOW': 'gold',
                                                'RED': 'red'
                                                },
                            color_discrete_sequence=px.colors.qualitative.G10)

    fig.update_layout(autosize=auto_size,
                      height=height,
                      width=width,
                      template=template)
    #fig.show()
    return fig
