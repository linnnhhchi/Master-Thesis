# %% Imports
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Conv1D, Dropout, Dense, Embedding, Flatten, RepeatVector, Concatenate
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD, RMSprop
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit
from keras import backend as K
import optuna
import time

start_time = time.time()

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

gpus = tf.config.list_physical_devices('GPU')
if len(gpus) > 1:
    try:
        # Restrict TensorFlow to only use GPU:0
        tf.config.set_visible_devices(gpus[1], 'GPU')
        print(f"Using only GPU: {gpus[1].name}")
    except RuntimeError as e:
        print(f"Error limiting GPUs: {e}")

# Allow memory growth on GPUs to prevent TensorFlow from allocating all GPU memory at once
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for all GPUs.")
    except RuntimeError as e:
        print(f"Error enabling memory growth: {e}")
        
# %% Load and preprocess data
df = pd.read_csv('/Users/linhchi/Downloads/Thesis /EV dataset/US/feature_data.csv')
df["date"] = pd.to_datetime(df["date"])
df.set_index('date', inplace=True)
df = df.loc[:'2021-12-10']

X = df.drop(columns=['charged_energy'])
y = df['charged_energy']

label_encoder = LabelEncoder()
X['land_types'] = label_encoder.fit_transform(X['land_types'])
X_land = np.array(X['land_types'], dtype=np.int32)
X = X.drop(columns=['land_types'])

numeric_features = ['temperature', 'year', 'dewpoint', 'road_density', 'commercial_density',
                    'residential_density', 'recreation_density', 'highway_proximity',
                    'public_transport_proximity', 'evcs_proximity', 'center_proximity',
                    'parking_density', 'hour_sin', 'hour_cos', 'month_cos', 'month_sin']
binary = list(set(X.columns) - set(numeric_features))

preprocessor = ColumnTransformer(transformers=[
    ('num', MinMaxScaler(), numeric_features),
    ('cat', OneHotEncoder(), binary)])

# Reset for cross-validation
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
tscv = TimeSeriesSplit(n_splits=5)
splits = list(tscv.split(X))

# %% Spatiotemporal LSTM
def objective(trial):
    lstm_units = trial.suggest_int("lstm_units", 32, 256, step=16)
    dropout_rate = trial.suggest_categorical("dropout", [0.2, 0.3, 0.5])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    two_layers = trial.suggest_categorical('two_layers', [True, False])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    val_losses = []

    for train_index, val_index in splits:
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        land_train_fold, land_val_fold = X_land[train_index], X_land[val_index]

        X_train_proc = preprocessor.fit_transform(X_train_fold, y_train_fold)
        X_val_proc = preprocessor.transform(X_val_fold)

        X_train_seq = X_train_proc.reshape((X_train_proc.shape[0], 1, X_train_proc.shape[1]))
        X_val_seq = X_val_proc.reshape((X_val_proc.shape[0], 1, X_val_proc.shape[1]))

        X_train_seq = np.array(X_train_seq, dtype=np.float32)
        X_val_seq = np.array(X_val_seq, dtype=np.float32)
        y_train_fold = np.array(y_train_fold, dtype=np.float32)
        y_val_fold = np.array(y_val_fold, dtype=np.float32)

        # Create optimizer fresh inside fold loop
        if optimizer_name == "adam":
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name == "sgd":
            optimizer = SGD(learning_rate=learning_rate)
        else:
            optimizer = RMSprop(learning_rate=learning_rate)
            
        with tf.device('/GPU:1'):
            seq_input = Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2]), name='seq_input')
            land_input = Input(shape=(1,), dtype='int32', name='land_input')

            embedding = Embedding(input_dim=np.max(X_land) + 1, output_dim=5)(land_input)
            embedding_flat = Flatten()(embedding)
            embedding_repeat = RepeatVector(X_train_seq.shape[1])(embedding_flat)

            merged = Concatenate()([seq_input, embedding_repeat])

            x = LSTM(lstm_units // 2, activation='relu', return_sequences=two_layers)(merged)
            x = Dropout(dropout_rate)(x)
            if two_layers:
                x = LSTM(lstm_units, activation='relu')(x)
                x = Dropout(dropout_rate)(x)

            output = Dense(1)(x)

            model = Model(inputs=[seq_input, land_input], outputs=output)
            model.compile(loss='mean_absolute_error', optimizer=optimizer)

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            model.fit([X_train_seq, land_train_fold], y_train_fold,
                    validation_data=([X_val_seq, land_val_fold], y_val_fold),
                    epochs=30, batch_size=batch_size, verbose=0,
                    callbacks=[early_stopping])

            val_loss = model.evaluate([X_val_seq, land_val_fold], y_val_fold, verbose=0)
            val_losses.append(val_loss)

        K.clear_session()

    return np.mean(val_losses)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
print("Best parameters:", study.best_params)


# %% Spatiotemporal CNN-LSTM
def objective(trial):
    filters = trial.suggest_int('filters',  32, 256, step=16)
    kernel_size = trial.suggest_categorical('kernel_size', [3, 4, 5])
    lstm_units = trial.suggest_categorical('lstm_units', [16, 32, 64, 96, 128])
    dropout_rate = trial.suggest_categorical('dropout', [0.2, 0.3, 0.5])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    optimizer_choice = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    val_losses = []

    for train_index, val_index in splits:
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        land_train_fold, land_val_fold = X_land[train_index], X_land[val_index]

        X_train_proc = preprocessor.fit_transform(X_train_fold, y_train_fold)
        X_val_proc = preprocessor.transform(X_val_fold)

        X_train_seq = X_train_proc.reshape((X_train_proc.shape[0], X_train_proc.shape[1], 1))
        X_val_seq = X_val_proc.reshape((X_val_proc.shape[0], X_val_proc.shape[1], 1))

        X_train_seq = np.array(X_train_seq, dtype=np.float32)
        X_val_seq = np.array(X_val_seq, dtype=np.float32)
        y_train_fold = np.array(y_train_fold, dtype=np.float32)
        y_val_fold = np.array(y_val_fold, dtype=np.float32)

        # Create optimizer inside the loop (every fold)
        if optimizer_choice == "adam":
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_choice == "sgd":
            optimizer = SGD(learning_rate=learning_rate)
        else:
            optimizer = RMSprop(learning_rate=learning_rate)

        with tf.device(device):
            seq_input = Input(shape=(X_train_seq.shape[1], 1), name='seq_input')
            land_input = Input(shape=(1,), dtype='int32', name='land_input')

            emb = Embedding(input_dim=np.max(X_land) + 1, output_dim=5)(land_input)
            emb = Flatten()(emb)
            emb = RepeatVector(X_train_seq.shape[1])(emb)

            x = Concatenate()([seq_input, emb])
            x = Conv1D(filters=filters // 2, kernel_size=kernel_size, activation='relu', padding='same')(x)
            x = Dropout(dropout_rate)(x)
            x = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same')(x)
            x = Dropout(dropout_rate)(x)
            x = LSTM(units=lstm_units, activation='relu')(x)
            output = Dense(1)(x)

            model = Model(inputs=[seq_input, land_input], outputs=output)
            model.compile(loss='mae', optimizer=optimizer)

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            model.fit([X_train_seq, land_train_fold], y_train_fold,
                      validation_data=([X_val_seq, land_val_fold], y_val_fold),
                      epochs=30, batch_size=batch_size, callbacks=[early_stopping], verbose=0)

            val_loss = model.evaluate([X_val_seq, land_val_fold], y_val_fold, verbose=0)
            val_losses.append(val_loss)

        K.clear_session()

    return np.mean(val_losses)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
print("Best parameters:", study.best_params)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total optimization time: {elapsed_time:.4f} seconds")


# %% Temporal LSTM
df = pd.read_csv('/content/feature_data.csv')
df["date"] = pd.to_datetime(df["date"])
df.set_index('date', inplace=True)
df = df.loc[:'2021-12-10']

X = df.drop(columns=['charged_energy', 'road_density', 'highway_proximity',
       'commercial_density', 'residential_density',
       'public_transport_proximity', 'evcs_proximity', 'parking_density',
       'recreation_density', 'center_proximity', 'land_types'])
y = df['charged_energy']

numeric_features = ['temperature', 'dewpoint', 'year', 'hour_sin', 'hour_cos',
       'month_sin', 'month_cos']
binary = ['day_of_week', 'weekend', 'holiday', 'covid', 'time_slot']

for col in binary:
    X[col] = X[col].astype(str)

preprocessor = ColumnTransformer(transformers=[
    ('num', MinMaxScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), binary)])

# Reset for cross-validation
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
tscv = TimeSeriesSplit(n_splits=3)
splits = list(tscv.split(X))

def objective(trial):
    lstm_units = trial.suggest_int("lstm_units", 32, 256, step=16)
    dropout_rate = trial.suggest_categorical("dropout", [0.2, 0.3, 0.5])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    two_layers = trial.suggest_categorical('two_layers', [True, False])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    val_losses = []

    for train_index, val_index in splits:
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        X_train_proc = preprocessor.fit_transform(X_train_fold, y_train_fold)
        X_val_proc = preprocessor.transform(X_val_fold)

        X_train_seq = X_train_proc.reshape((X_train_proc.shape[0], 1, X_train_proc.shape[1]))
        X_val_seq = X_val_proc.reshape((X_val_proc.shape[0], 1, X_val_proc.shape[1]))

        X_train_seq = np.array(X_train_seq, dtype=np.float32)
        X_val_seq = np.array(X_val_seq, dtype=np.float32)
        y_train_fold = np.array(y_train_fold, dtype=np.float32)
        y_val_fold = np.array(y_val_fold, dtype=np.float32)

        # Select optimizer
        if optimizer_name == "adam":
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name == "sgd":
            optimizer = SGD(learning_rate=learning_rate)
        else:
            optimizer = RMSprop(learning_rate=learning_rate)

        with tf.device(device):
            model = Sequential()
            model.add(Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
            model.add(LSTM(lstm_units//2, activation='relu', return_sequences=two_layers))
            model.add(Dropout(dropout_rate))

            if two_layers:
                model.add(LSTM(lstm_units, activation='relu'))
                model.add(Dropout(dropout_rate))

            model.add(Dense(1))

            model.compile(loss='mean_absolute_error', optimizer=optimizer)
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            model.fit(X_train_seq, y_train_fold,
                      validation_data=(X_val_seq, y_val_fold),
                      epochs=30, batch_size=batch_size,
                      verbose=0, callbacks=[early_stopping])
            
            val_loss = model.evaluate(X_val_seq, y_val_fold, verbose=0)
            val_losses.append(val_loss)

        K.clear_session()
    return np.mean(val_losses)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
print("Best parameters:", study.best_params)



# %% Temporal CNN-LSTM
df = pd.read_csv('feature_data.csv')
df["date"] = pd.to_datetime(df["date"])
df.set_index('date', inplace=True)
df = df.loc[:'2021-12-10']

X = df.drop(columns=['charged_energy', 'road_density', 'highway_proximity',
       'commercial_density', 'residential_density',
       'public_transport_proximity', 'evcs_proximity', 'parking_density',
       'recreation_density', 'center_proximity', 'land_types'])
y = df['charged_energy']

numeric_features = ['temperature', 'dewpoint', 'year', 'hour_sin', 'hour_cos',
       'month_sin', 'month_cos']
binary = ['day_of_week', 'weekend', 'holiday', 'covid', 'time_slot']

for col in binary:
    X[col] = X[col].astype(str)

preprocessor = ColumnTransformer(transformers=[
    ('num', MinMaxScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), binary)])

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
tscv = TimeSeriesSplit(n_splits=3)
splits = list(tscv.split(X))

def objective(trial):
    filters = trial.suggest_int('filters', 32, 256, step=16)
    kernel_size = trial.suggest_categorical('kernel_size', [3, 4, 5])
    lstm_units = trial.suggest_categorical('lstm_units', [16, 32, 64, 96, 128])
    dropout_rate = trial.suggest_categorical('dropout', [0.2, 0.3, 0.5])
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    optimizer_choice = trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop'])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    val_losses = []

    for train_index, val_index in splits:
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        X_train_proc = preprocessor.fit_transform(X_train_fold)
        X_val_proc = preprocessor.transform(X_val_fold)

        X_train_seq = X_train_proc.reshape((X_train_proc.shape[0], X_train_proc.shape[1], 1))
        X_val_seq = X_val_proc.reshape((X_val_proc.shape[0], X_val_proc.shape[1], 1))

        X_train_seq = np.array(X_train_seq, dtype=np.float32)
        X_val_seq = np.array(X_val_seq, dtype=np.float32)
        y_train_fold = np.array(y_train_fold, dtype=np.float32)
        y_val_fold = np.array(y_val_fold, dtype=np.float32)

        # Choose optimizer
        if optimizer_choice == "adam":
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_choice == "sgd":
            optimizer = SGD(learning_rate=learning_rate)
        else:
            optimizer = RMSprop(learning_rate=learning_rate)

        # Build model
        with tf.device('/GPU:1'):  
            model = Sequential()
            model.add(Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
            model.add(Conv1D(filters=filters//2, kernel_size=kernel_size, activation='relu', padding='same'))
            model.add(Dropout(rate=dropout_rate))
            model.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='same'))
            model.add(Dropout(rate=dropout_rate))
            model.add(LSTM(units=lstm_units, activation='relu', return_sequences=False))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1))
            model.compile(loss='mae', optimizer=optimizer)

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            model.fit(X_train_seq, y_train_fold,
                      validation_data=(X_val_seq, y_val_fold),
                      epochs=30, batch_size=batch_size,
                      callbacks=[early_stopping], verbose=0)

            val_loss = model.evaluate(X_val_seq, y_val_fold, verbose=0)
            val_losses.append(val_loss)

        K.clear_session()

    return np.mean(val_losses)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
print("Best parameters:", study.best_params)

elapsed_time = time.time() - start_time
print(f"Total optimization time: {elapsed_time:.2f} seconds")