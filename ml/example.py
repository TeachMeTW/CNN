# 1
print("1")
import os
os.environ['NUMEXPR_MAX_THREADS'] = '96'
import glob
import joblib
import logging
import warnings

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)
warnings.filterwarnings('ignore')

# 2
print("2")
# imports for notebook
import pandas as pd
import numpy as np
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
pd.set_option('display.max_columns', None) # for notebook only
pd.set_option('display.max_row', None) # for notebook only
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.offline
init_notebook_mode(connected=True)
import plotly.io as pio
pio.renderers.default = 'notebook' #

# 3
print("3")
# preprocessing
from sklearn import preprocessing
from sklearn import metrics
from datetime import datetime
TEMPLATE = 'ggplot2'

import tensorflow as tf
import keras

print("3.5")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# 4
print("4")
import kagglehub

# Download latest version
path = kagglehub.dataset_download("garystafford/environmental-sensor-data-132k")

print(f'Path to dataset files: {path}')

# 5
print("5")
file_path = path+'/iot_telemetry_data.csv'
df=pd.read_csv(file_path)
print(df.shape)
df.head()

# 6
print("6")
df.info()

# 7
print("7")
b = df.select_dtypes(include='bool').columns.to_list()
df[b] = df[b].astype(float)
df.head()

# 8
print("8")
MACHINE_TYPE = 'mac'
ANALYSIS_NAME = 'IoT_calPoly_example_{}'
RANDOM_STATE = None
OPTIMIZATION_TRIALS = 4
SCALER = 'Robust'
ERROR_CUTOFF = 99.97
# sample percentage for training data
SAMPLE_PERC = 0.8

# 9
print("9")
ID_COL = 'device'
TIMESTAMP = 'ts'
CONTEXT = [TIMESTAMP, ID_COL]
FEATURES = [f for f in df.columns if f not in CONTEXT]
#FEATURES = [f for f in FEATURES if f not in b]
print('CONTEXT:\n# Context: {}\n\n{}\n\nFEATURES:\n# Features: {}\n\n{}'.format(len(CONTEXT),CONTEXT,len(FEATURES),FEATURES))

# 10
print("10")
df[ID_COL] = df[ID_COL].astype(str)
df[FEATURES] = df[FEATURES].astype(float)
#df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP],errors='coerce')
# 11 
print("11")
df = df.sort_values([TIMESTAMP,ID_COL],ascending=True).reset_index(drop=True)

df.groupby(ID_COL)[FEATURES].describe().T

# 12
print("12")
MODEL_TIME = str(datetime.now())[:10]
MODEL_NAME = ANALYSIS_NAME.format(f'/model/{ANALYSIS_NAME.format(MODEL_TIME)}_best_weights.hdf5')
SCALER_NAME = ANALYSIS_NAME.format(f'/model/{ANALYSIS_NAME.format(MODEL_TIME)}_scaler.save')
LOG_DIR = ANALYSIS_NAME.format('/logs')
PLOT_DIR = ANALYSIS_NAME.format('/context_plots')
MODEL_DIR = ANALYSIS_NAME.format('/model')

ll = [LOG_DIR, PLOT_DIR, MODEL_DIR]
for l in ll:
    if not os.path.exists(l):
        os.makedirs(l)

print('Model Save Path: {}\nScaler Save Path: {}\nPlot Directory: {}\nLog Directory: {}'.format(MODEL_NAME,
                                                                                      SCALER_NAME,
                                                                                      PLOT_DIR,
                                                                                      LOG_DIR))

# 13
print("13")
stable_devices = ['00:0f:00:70:91:0a',
                  'b8:27:eb:bf:9d:51']
train = df[df[ID_COL].isin(stable_devices)].sample(frac=SAMPLE_PERC,
                                                   random_state=RANDOM_STATE)
df_copy = df.copy()
print('Training Data Shape: {}\nTraining IDs: {}'.format(train[FEATURES].shape,
                                                                train[ID_COL].unique()
                                                        )
     )
# 14
print("14")
import AE_tools

train, scaler = AE_tools.prep_data(data_frame=train,
                                   features=FEATURES,
                                   scaler=SCALER)
print(train[FEATURES].shape)
train[FEATURES].head()

print("15")
# save scaler
joblib.dump(scaler, SCALER_NAME)
print("16")
from AE import AutoEncoder

dd1 = datetime.now()
ae = AutoEncoder(data=train,
                 features=FEATURES,
                 model_name=MODEL_NAME,
                 log_dir=LOG_DIR,
                 random_state=RANDOM_STATE,
                 machine_type=MACHINE_TYPE,
                 n_trials = OPTIMIZATION_TRIALS).fit_pipeline()
dd2 = datetime.now()
runtime = dd2-dd1
print("17")
print(f'Build / Optimize Runtime: {runtime}')
print("18")
autoencoder = ae['autoencoder']
df_history = ae['history']
best_params = ae['best_params']
print('Model Loss: {}'.format(ae['model_loss']))
best_params
print("19")
autoencoder.summary()
print("20")
loss_save_str = ANALYSIS_NAME.format(f'/model/{ANALYSIS_NAME.format(MODEL_TIME)}_model_loss.html')
fig = AE_tools.plot_loss(history=df_history,
                         title="Train vs Validation Loss",
                         template=TEMPLATE)
fig.write_html(loss_save_str)
#fig.show()
print("21")
# load model
autoencoder = tf.keras.models.load_model(MODEL_NAME)
print("22")
autoencoder.summary()
print("23")
# load scaler
scaler = joblib.load(SCALER_NAME)
df[FEATURES] = scaler.transform(df[FEATURES])
print("24")
latent_space = AE_tools.build_latent_space(auto_encoder=autoencoder,
                                           data_frame=df,
                                           features=FEATURES)
print("25")
latent_space_names = latent_space.columns.to_list()
df_copy[latent_space_names] = latent_space[latent_space_names]
print("26")
train, ERROR_THRESHOLD = AE_tools.calculate_error_threshold(auto_encoder=autoencoder,
                                                            train_data=train,
                                                            features=FEATURES,
                                                            percentile=ERROR_CUTOFF)
print(f'ANOMALY THRESHOLD: {round(ERROR_THRESHOLD,3)}')
print("27")
df, preds = AE_tools.calculate_error(auto_encoder=autoencoder,
                                     data_frame=df,
                                     features=FEATURES)
print(len(preds))
df_copy['reconstruction_error'] = df['reconstruction_error']
print("28")
preds_stats = preds[FEATURES].describe().T.sort_values('mean', ascending=False).reset_index()
preds_stats = preds_stats.rename(columns={'index':'sensor'})
print(preds_stats[:10])
print("29")
anomalies = df_copy[df_copy['reconstruction_error'] > ERROR_THRESHOLD]
print('N Anomalous Part IDs: {}\nTotal Unique Part IDs: {}'.format(anomalies[ID_COL].nunique(),
                                                                     df_copy[ID_COL].nunique()))
print("30")
print('% Anomalous Records: {}'.format(
    round((anomalies.shape[0]/df.shape[0])*100,
          2)))
print("31")
print(CONTEXT)
print("32")
for color in CONTEXT[1:]:
    fig = AE_tools.plot_latent_space(data_frame=df_copy,
                                     color=color,
                                     hover_data=CONTEXT,
                                     title=f'Latent Space: {color} Overlay',
                                     template=TEMPLATE)
    fig.write_html(f'{PLOT_DIR}/latent_space_{color}_overlay.html')
    #fig.show()
print("33")
sensor_by_mean_error = list(preds_stats['sensor'].reset_index(drop=True))
print(sensor_by_mean_error)
print("34")
print("35")
for color in sensor_by_mean_error:
    fig = AE_tools.plot_latent_space(data_frame=df_copy,
                                     color=color,
                                     hover_data=CONTEXT,
                                     title=f'Latent Space: {color} Overlay',
                                     template=TEMPLATE)
    fig.write_html(f'{PLOT_DIR}/latent_space_{color}_overlay.html')
    #fig.show()
print("36")
for color in b:
    fig = AE_tools.plot_latent_space(data_frame=df_copy,
                                     color=color,
                                     hover_data=CONTEXT,
                                     title=f'Latent Space: {color} Overlay',
                                     template=TEMPLATE)
    fig.write_html(f'{PLOT_DIR}/latent_space_{color}_overlay.html')
    #fig.show()
print("37")
overlay_color = 'reconstruction_error'#sensor_by_mean_error[0]
df_copy[TIMESTAMP] = df_copy[TIMESTAMP].astype(str)
sns = list(df[ID_COL].unique())
for sn in sns:
    hh = df_copy[df_copy[ID_COL]==sn].reset_index(drop=True)
    #tb = list(hh[TEST_BED].unique())[0]
    title=f'Reconstruction Error / Time: {sn}'
    fig = AE_tools.plot_error_over_time(df=hh,
                                  timestamp_col=TIMESTAMP,
                                  id_col=ID_COL,
                                  title=title,
                                  template=TEMPLATE,
                                  color_overlay=overlay_color,
                                  error_threshold=ERROR_THRESHOLD)
    fig.write_html(f'{PLOT_DIR}/error_over_time_{sn}_{overlay_color}_overlay.html')
    #fig.show()
print("38")
reconstructed_anomalies_error = [AE_tools.compute_error_per_dim(point=x,
                                                                input_dim=len(FEATURES),
                                                                df=df[FEATURES],
                                                                reconstructions=preds) for x in anomalies.index]

reconstructed_anomalies_error = pd.DataFrame(reconstructed_anomalies_error,
                                             columns=FEATURES,
                                             index=anomalies.index)

TITLE = 'Mean Reconstruction Error by Feature for All Anomalies'

fig = AE_tools.plot_mean_error(error_frame=reconstructed_anomalies_error,
                               template=TEMPLATE,
                               title=TITLE)
fig.write_html(f'{PLOT_DIR}/mean_error_by_feature_anomalies.html')
#fig.show()
print("39")
TITLE = 'Device {} Reconstruction Error at Time {}'

list_len = 5
anomaly_idx_list = list(anomalies.index[:list_len])

for idx in anomaly_idx_list:
    hh = pd.DataFrame(reconstructed_anomalies_error.loc[[idx]].T).reset_index(drop=False)
    print(hh.shape)
    dd = df_copy.loc[[idx]]
    context = dd[CONTEXT].reset_index(drop=True)
    print(f'CONTEXT:\n{context}\n')
    hh.columns = ['sensor','reconstruction_error']
    ts = context.loc[[0]]['ts'].values[0]
    i = context.loc[[0]]['device'].values[0]
    TITLE = TITLE.format(i,ts)
    fig = AE_tools.plot_error_per_feature(error_record=hh,
                                    title=TITLE,
                                    template=TEMPLATE,
                                    auto_size=True)
    fig.write_html(f'{PLOT_DIR}/error_by_feature_{idx}_{CONTEXT[0]}_{CONTEXT[1]}_overlay.html')
    #fig.show()
