from flask import Flask, request, jsonify
import os
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import MLP, NBEATS
from neuralforecast.auto import AutoTFT
from neuralforecast.losses.pytorch import MAE
from neuralforecast.losses.numpy import mae
from ray import tune
import tempfile
import json
import xgboost as xg
from sklearn.preprocessing import StandardScaler
import pickle
from pickle import load, dump

app = Flask(__name__)

def create_model_configs(horizon):
    nbeats1 = NBEATS(h=horizon, input_size=36, max_steps=10, early_stop_patience_steps=5, batch_size=24, windows_batch_size=24)
    nbeats2 = NBEATS(h=horizon, input_size=28, max_steps=10, early_stop_patience_steps=5, batch_size=14, windows_batch_size=28)
    mlp = MLP(h=horizon, input_size=36, max_steps=10, early_stop_patience_steps=5, batch_size=24, windows_batch_size=24)
    config = {
        "input_size": tune.choice([horizon]),
        "hidden_size": tune.choice([32]),
        "n_head": tune.choice([4]),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "scaler_type": tune.choice(['robust']),
        "max_steps": tune.choice([10]),
        "windows_batch_size": tune.choice([32]),
        "check_val_every_n_epoch": tune.choice([10]),
        "random_seed": tune.randint(1, 20),
    }
    autotft = AutoTFT(h=horizon, loss=MAE(), config=config, num_samples=5)
    return [nbeats1, nbeats2, mlp, autotft]

def preprocess_dataframe(file_path, SKU):
    df = pd.read_csv(file_path)
    df_tmp = pd.DataFrame(df)
    df_tmp['unique_id'] = SKU
    cols = ['unique_id'] + [col for col in df_tmp.columns if col != 'unique_id']
    df_tmp = df_tmp[cols]
    unnamed_cols = df.columns[df.columns.str.startswith('Unnamed:')]
    df_tmp.rename(columns={'date': 'ds', 'final_value': 'y'}, inplace=True)
    df_tmp = df_tmp.drop(unnamed_cols, axis=1)
    df_tmp['ds'] = pd.to_datetime(df_tmp['ds'])
    df_tmp.set_index('ds', inplace=True)
    # df_tmp = df_tmp[~df_tmp.index.duplicated(keep='first')] 
    full_date_range = pd.date_range(start=df_tmp.index.min(), end=df_tmp.index.max(), freq='D')
    df_complete = df_tmp.reindex(full_date_range)
    df_complete.reset_index(inplace=True)
    df_complete.rename(columns={'index': 'ds'}, inplace=True)
    df_complete['unique_id'] = df_complete['unique_id'].ffill()
    df_complete['y'] = df_complete['y'].fillna(0)
    return df_complete

@app.route('/train', methods=['POST'])
def train_models():
    if 'config' not in request.files:
        return jsonify({"error": "No config file provided"}), 400

    config_file = request.files['config']
    config = json.load(config_file)
    horizon = config.get("horizon", 7)
    models = config.get("models", ["AutoTFT", "MLP", "NBEATS", "NBEATS1"])

    if not models:
        return jsonify({"error": "No models specified in config"}), 400

    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist('files')
    list_sku = [file.filename.split('_filter')[0] for file in files]
    list_df = []
    list_nf = []
    list_BLPreds = []
    performance_metrics = {}

    model_configs = create_model_configs(horizon)

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file.save(temp_file.name)
            SKU = file.filename.split('_filter')[0]
            df_complete = preprocess_dataframe(temp_file.name, SKU)
            list_df.append(df_complete)
            Y_train_df = df_complete[df_complete.ds < df_complete['ds'].values[-horizon]]
            Y_test_df = df_complete[df_complete.ds >= df_complete['ds'].values[-horizon]].reset_index(drop=True)

            nf = NeuralForecast(models=model_configs, freq='D')
            nf.fit(df=Y_train_df, val_size=horizon)
            print('\n\n\n')
            print("Current Path: ", os.getcwd())
            print('\n\n\n')
            path=f'.\\checkpoints\\{SKU}\\'
            print("Save Path: ", path)
            print('\n\n\n')
            test_path=f'.\\checkpoints\\{SKU}\\'
            print("Test Path: ", test_path)
            print('\n\n\n')
            nf.save(path, model_index=None, overwrite=True, save_dataset=True)
            list_nf.append(nf)

            print(Y_test_df)

            Y_test_df_tmp = Y_test_df
            Y_test_df_tmp = Y_test_df_tmp.set_index('unique_id')
            print(Y_test_df_tmp)

            forecasts = nf.predict(futr_df=Y_test_df)
            print(models)
            print(forecasts)
            for model_name in models:
                Y_test_df_tmp[model_name] = forecasts[model_name]
                performance_metrics.setdefault(SKU, {})[model_name] = mae(Y_test_df_tmp['y'], Y_test_df_tmp[model_name])
            
            print(Y_test_df_tmp)
            print(Y_test_df)

            forecasts['y'] = Y_test_df_tmp['y'] 
            print(Y_test_df_tmp['y'])
            print(forecasts)
            list_BLPreds.append(forecasts)
            
        # list_BLPreds = []
        # for SKU in list_sku:
        #     # pred_path = f'api\\{SKU}_filter_3_predict.csv'
        #     # df = pd.read_csv(pred_path)
        #     # df['ds'] = pd.to_datetime(df['ds'])
        #     # df = df.drop(['Unnamed: 0'], axis=1)
        #     # nf = NeuralForecast.load(path=f'\\checkpoints\\{SKU}\\')
        #     forecasts = nf.predict(futr_df=Y_test_df)
        #     forecasts['y'] = Y_test_df_tmp['y'] 
        #     print(Y_test_df_tmp['y'])
        #     print(forecasts)
        #     list_BLPreds.append(forecasts)

    for i, SKU in enumerate(list_sku):
        dataset = list_BLPreds[i].reset_index()
        # print(dataset)
        dataset = dataset.drop(['unique_id', 'ds'], axis=1)
        print(dataset)
        X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
        split_point = int(len(X) * 0.8)
        train_X, test_X = X[:split_point], X[split_point:]
        train_y, test_y = y[:split_point], y[split_point:]
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        print(train_X)
        scaled_train_X = x_scaler.fit_transform(train_X)
        scaled_test_X = x_scaler.transform(test_X)
        dump(x_scaler, open(f'x_scaler_{SKU}.pkl', 'wb'))
        scaled_train_y = y_scaler.fit_transform(train_y.values.reshape(-1, 1))
        scaled_test_y = y_scaler.transform(test_y.values.reshape(-1, 1))
        dump(y_scaler, open(f'y_scaler_{SKU}.pkl', 'wb'))
        xgb_r = xg.XGBRegressor(objective='reg:linear', n_estimators=100000, seed=123)
        xgb_r.fit(scaled_train_X, scaled_train_y)
        with open(f'xgb_model_{SKU}.pkl', 'wb') as f:
            pickle.dump(xgb_r, f)
        scaled_predictions = xgb_r.predict(scaled_test_X)
        predictions = y_scaler.inverse_transform(scaled_predictions.reshape(-1, 1))
        print(predictions)
        performance_metrics[SKU]['MetaLearner'] = mae(test_y.values, predictions.ravel())

    return jsonify(performance_metrics)

@app.route('/inference', methods=['POST'])
def get_final_predictions():
    print(os.getcwd())
    if 'config' not in request.files:
        return jsonify({"error": "No config file provided"}), 400

    config_file = request.files['config']
    config = json.load(config_file)
    horizon = config.get("horizon", 7)
    models = config.get("models", ["AutoTFT", "MLP", "NBEATS", "NBEATS1"])

    if not models:
        return jsonify({"error": "No models specified in config"}), 400

    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist('files')
    results = {}

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file.save(temp_file.name)
            SKU = file.filename.split('_filter')[0]
            df = preprocess_dataframe(temp_file.name, SKU)
            # nf = NeuralForecast.load(path=f'./checkpoints/{SKU}/')
            load_path = f'.\\checkpoints\\{SKU}\\'
            # load_path = 'C:\checkpoints\V240-01-001-B'
            print(load_path)
            nf = NeuralForecast.load(path = load_path)
            forecasts = nf.predict(df=df)
            forecasts.reset_index(inplace=True)
            forecasts = forecasts.drop(['unique_id', 'ds'], axis=1, errors='ignore')
            forecasts.columns = ["NBEATS", "NBEATS1", "MLP", "AutoTFT"]
            print(forecasts)
            x_scaler = load(open(f'x_scaler_{SKU}.pkl', 'rb'))
            y_scaler = load(open(f'y_scaler_{SKU}.pkl', 'rb'))
            print(forecasts)
            scaled_forecasts = x_scaler.transform(forecasts)
            with open(f'xgb_model_{SKU}.pkl', 'rb') as f:
                meta_learner = pickle.load(f)
            scaled_predictions = meta_learner.predict(scaled_forecasts)
            predictions = y_scaler.inverse_transform(scaled_predictions.reshape(-1, 1))
            results[SKU] = predictions.tolist()

    return jsonify(results)

@app.route('/test', methods=['GET', 'POST'])
def test_api():
    return "API is Running"

if __name__ == '__main__':
    app.run(debug=True)
