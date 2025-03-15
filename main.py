import os
import warnings
import argparse
import functools

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument("--seed", type = int, default = 2024)
    parser.add_argument("--gpu", default = None, help='To run on gpu, enter the gpu device number. (ex: 0)')
    
    parser.add_argument("--pm", type = float, default = 10, help = "Select in (1, 2.5, 10)")
    parser.add_argument("--window_size", type = int, default = 360, help = "(15, 60, 360, 720, 1440) are recommended")
    parser.add_argument("--max_sensor_scale", type = int, default = 300)
    
    parser.add_argument("--epoch", type = int, default = 10)
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--learning_rate", type = float, default = 1e-4)
    parser.add_argument("--shuffle_sensor_size", type = int, default = 4, help = "The number of shuffle size between sensors during training.")
    
    parser.add_argument("--save_path", default = "./model", help = "Training or evaluate model path.")
    
    config = parser.parse_args()
    return config

def read_data(data_list, path = False, label_smooth = 0, **kwargs):
    data_list = [data_list] if not isinstance(data_list, (tuple, list)) else data_list
    
    xs, ys = [], []
    for p in data_list:
        if not path:
            df = pd.read_csv(p)
            df['date'] = pd.to_datetime(df['date'])
            #df['week'] = df['date'].dt.isocalendar().week
            df['day'] = df['date'].dt.day
            df['day_of_week'] = df['date'].dt.dayofweek
            df['hour'] = df['date'].dt.hour
            df['minute'] = df['date'].dt.minute
            x, y = lbt.utils.data_util.get_window_dataset(df, label_smooth = label_smooth, **kwargs)
            ys.append(y)
        else:
            x = p
        xs.append(x)
    xs = np.vstack(xs)
    if path:
        return xs
    else:
        ys = np.vstack(ys)
        return xs, ys

def load(x, shuffle = True, max_scale = 300, **kwargs):
    x = [_x[0].decode("utf-8") for _x in x]
    xs, ys = read_data(x, **kwargs)
    ys = ys / max_scale
    
    if shuffle:
        indices = np.arange(len(xs))
        np.random.shuffle(indices)
    xs = xs[indices]
    ys = ys[indices]
    xs = tf.convert_to_tensor(xs, dtype = tf.float32)
    ys = tf.convert_to_tensor(ys, dtype = tf.float32)
    return xs, ys

def scale_rmse(x, y, scale = 1):
    return tf.sqrt(tf.reduce_mean((x * scale - y * scale) ** 2))

if __name__ == "__main__":
    config = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu) if config.gpu is not None else "-1"
    
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import matplotlib.pyplot as plt
    
    import lbt
    import lbt.datasets.senseurcity.metadata as meta
    
    pm_map = {10.0:0, 2.5:1, 1.0:2}
    pm_invert_map = {0:10, 1:2.5, 2:1}
    DATA_PARAMS = {'X_col': ["5310CAT", "5325CAT", "5301CAT"][pm_map[float(config.pm)]], 
                   'y_col': ["OPCN3PM10", "OPCN3PM25", "OPCN3PM1"][pm_map[float(config.pm)]], 
                   'window_length': config.window_size, 
                   'target_window_length': 1,
                   'accept_window_interval': None,
                   'temporal_info': ['day', 'day_of_week', 'hour', 'minute']}
    load_func = functools.partial(load, shuffle = True, max_scale = config.max_sensor_scale, **DATA_PARAMS)
    def rmse(x, y, scale = config.max_sensor_scale):
        return scale_rmse(x, y, scale = scale)
    
    rmse_list = []
    for idx, (data_type, data_list) in enumerate([(meta.ANTWERP, meta.ANTWERP_FIRST_COL_LIST),
                                                   (meta.OSLO, meta.OSLO_FIRST_COL_LIST), 
                                                   (meta.ZAGREB, meta.ZAGREB_FIRST_COL_LIST)]):
        print("\n\n")
        print("#" * 50)
        print("#" * 15 + "{0}  (PM:{1})".format({data_type[:-1]}, pm_invert_map[pm_map[float(config.pm)]]) + "#" * 15)
        print("#" * 50)
        
        lbt.utils.random.set_seed(config.seed, determinism = True)
        csv_list = ['{0}/{1}{2}_FirstCol.csv'.format(meta.FDCOL_DATASET, data_type, name) for name in data_list]
        data_list = ['./dataset/{0}{1}'.format(data_type, name) for name in data_list]
        
        X_train = read_data(csv_list[:-2], path = True, **DATA_PARAMS)
        X_valid, y_valid = read_data(csv_list[-2], **DATA_PARAMS)
        X_test, y_test = read_data(csv_list[-1], **DATA_PARAMS)
        
        train_dataset = tf.data.Dataset.from_tensor_slices(X_train)
        train_dataset = train_dataset.shuffle(train_dataset.cardinality(), seed = config.seed, reshuffle_each_iteration = True)
        train_dataset = train_dataset.batch(config.shuffle_sensor_size)
        train_dataset = train_dataset.map(lambda x: tf.numpy_function(load_func, [x], (tf.float32, tf.float32)))
        train_dataset = train_dataset.unbatch()
        train_dataset = train_dataset.batch(config.batch_size, drop_remainder = False)
        train_dataset = train_dataset.map(lambda x, y: (tf.reshape(x, [-1, config.window_size, 5]), tf.reshape(y, [-1, 1])))
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid / config.max_sensor_scale)).batch(config.batch_size * 4, drop_remainder = False)
        valid_dataset = valid_dataset.map(lambda x, y: (tf.reshape(x, [-1, config.window_size, 5]), tf.reshape(y, [-1, 1])))
        
        valid_x_dataset = tf.data.Dataset.from_tensor_slices(X_valid).batch(config.batch_size * 4, drop_remainder = False)
        valid_x_dataset = valid_x_dataset.map(lambda x: tf.reshape(x, [-1, config.window_size, 5]))
        test_dataset = tf.data.Dataset.from_tensor_slices(X_test).batch(config.batch_size * 4, drop_remainder = False)
        test_dataset = test_dataset.map(lambda x: tf.reshape(x, [-1, config.window_size, 5]))
        
        x = tf.keras.layers.Input(shape = [config.window_size, 5])
        out = lbt.layers.Normalizer(shift = 0, scale = config.max_sensor_scale)(x)
        out = lbt.models.Tesla()(out)
        model = tf.keras.Model(x, out)
        
        model_path = "{0}/{1}pm{2}_w{3}.h5".format(config.save_path, data_type, pm_invert_map[pm_map[float(config.pm)]], config.window_size)
        os.makedirs(os.path.dirname(model_path), exist_ok = True)

        if not os.path.exists(model_path):
            logger = tf.keras.callbacks.CSVLogger("{0}/{1}pm{2}_w{3}.log".format(config.save_path, data_type, pm_invert_map[pm_map[float(config.pm)]], config.window_size))
            checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = model_path, save_best_only = True, monitor = "val_rmse")

            optimizer = tf.keras.optimizers.Adam(learning_rate = config.learning_rate)

            model.compile(optimizer = optimizer, loss='mean_squared_error', metrics = [rmse])
            model.fit(train_dataset, validation_data = valid_dataset, epochs = config.epoch, callbacks = [logger, checkpoint], verbose = 1)
        model.load_weights(model_path)
        
        pred_val = model.predict(valid_x_dataset, verbose = 0) * config.max_sensor_scale
        pred_test = model.predict(test_dataset, verbose = 0) * config.max_sensor_scale
        
        rmse_val, rmse_test = scale_rmse(pred_val, y_valid, scale = 1).numpy(), scale_rmse(pred_test, y_test, scale = 1).numpy()
        print("val rmse : {0}, test rmse : {1}".format(rmse_val, rmse_test))
        rmse_list.append(rmse_test)
        
        plt.ioff()
        plt.figure(figsize=(10, 4))
        plt.plot(range(len(pred_test)), X_test[:, -1, 0], label = 'Raw input', alpha = 0.5, color = "gray")
        plt.plot(range(len(pred_test)), y_test, label = 'Reference', alpha = 0.7, color = "orange")
        plt.plot(range(len(pred_test)), pred_test, label = 'Pred', alpha = 1., color = "red")
        plt.xlim(0, len(pred_test))
        plt.ylim(0, 300)
        plt.xlabel('Sequence')
        plt.ylabel('Concentration $(\mu g/m^3)$')
        plt.legend(loc = "upper right")
        plt.savefig("{0}/{1}pm{2}_w{3}.png".format(config.save_path, data_type, pm_invert_map[pm_map[float(config.pm)]], config.window_size))
    
if len(rmse_list) > 0:
    print("avg test rmse: ", sum(rmse_list) / len(rmse_list), rmse_list)