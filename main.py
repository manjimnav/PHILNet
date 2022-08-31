import os
import argparse
import json
import itertools
import time
import requests
from multiprocessing import Process, Manager
import numpy as np
import random
import pandas as pd
from metrics import METRICS, evaluate
from preprocessing import denormalize
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

def notify_slack(msg, webhook=None):
    if webhook is None:
        webhook = os.environ.get("webhook_slack")
    if webhook is not None:
        try:
            requests.post(webhook, json.dumps({"text": msg}))
        except:
            print("Error while notifying slack")
            print(msg)
    else:
        print("NO WEBHOOK FOUND")


def check_params(datasets, models, results_path, parameters, metrics, csv_filename):
    assert len(datasets) > 0, "dataset parameter is not well defined."
    assert all(
        os.path.exists(ds_path) for ds_path in datasets
    ), "dataset paths are not well defined."
    assert all(
        param in parameters.keys()
        for param in [
            "normalization_method",
            "past_history_factor",
            "batch_size",
            "epochs",
            "max_steps_per_epoch",
            "learning_rate",
            "model_params",
        ]
    ), "Some parameters are missing in the parameters file."
    assert all(
        model in parameters["model_params"] for model in models
    ), "models parameter is not well defined."
    assert metrics is None or all(m in METRICS.keys() for m in metrics)


def read_results_file(csv_filepath, metrics):
    try:
        results = pd.read_csv(csv_filepath, sep=";", index_col=0)
    except IOError:
        results = pd.DataFrame(
            columns=[
                "DATASET",
                "MODEL",
                "MODEL_INDEX",
                "MODEL_DESCRIPTION",
                "FORECAST_HORIZON",
                "PAST_HISTORY_FACTOR",
                "PAST_HISTORY",
                "BATCH_SIZE",
                "EPOCHS",
                "STEPS",
                "OPTIMIZER",
                "LEARNING_RATE",
                "NORMALIZATION",
                "TEST_TIME",
                "TRAINING_TIME",
                *metrics,
                "LOSS",
                "VAL_LOSS",
                "REDUCE_DATA"
            ]
        )
    return results


def product(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def read_data(dataset_path, normalization_method, past_history_factor, reduce_data):
    # read normalization params
    norm_params = None
    with open(
        os.path.normpath(dataset_path)
        + "/{}/norm_params.json".format(normalization_method),
        "r",
    ) as read_file:
        norm_params = json.load(read_file)

    # read training / validation data
    tmp_data_path = os.path.normpath(dataset_path) + "/{}/{}/".format(
        normalization_method, past_history_factor
    )

    x_train = np.load(tmp_data_path + "x_train.np.npy")
    y_train = np.load(tmp_data_path + "y_train.np.npy")
    x_test = np.load(tmp_data_path + "x_test.np.npy")
    y_test = np.load(tmp_data_path + "y_test.np.npy")
    y_test_denorm = np.asarray(
        [
            denormalize(y_test[i], norm_params[i], normalization_method)
            for i in range(y_test.shape[0])
        ]
    )
    
    print(reduce_data)
    if reduce_data is not None:
        
        x_train = x_train[-int(x_train.shape[0]*reduce_data):, :]
        y_train = y_train[-int(y_train.shape[0]*reduce_data):, :]
        #x_test = x_test[-int(x_test.shape[0]*reduce_data):, :]
        #y_test = y_test[-int(y_test.shape[0]*reduce_data):, :]
        #y_test_denorm = y_test_denorm[-int(y_test_denorm.shape[0]*reduce_data):, :]
        #norm_params = norm_params[-int(y_test_denorm.shape[0]*reduce_data):]
        
        
    print("TRAINING DATA")
    print("Input shape", x_train.shape)
    print("Output_shape", y_train.shape)
    print("TEST DATA")
    print("Input shape", x_test.shape)
    print("Output_shape", y_test.shape)

    return x_train, y_train, x_test, y_test, y_test_denorm, norm_params


def _run_experiment(
    gpu_device,
    dataset,
    dataset_path,
    results_path,
    csv_filepath,
    metrics,
    epochs,
    normalization_method,
    past_history_factor,
    max_steps_per_epoch,
    batch_size,
    learning_rate,
    model_name,
    model_index,
    model_args,
    reduce_data
):
    import gc
    import tensorflow as tf
    from models import create_model

    tf.keras.backend.clear_session()

    def select_gpu_device(gpu_number):
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if len(gpus) >= 2 and gpu_number is not None:
            device = gpus[gpu_number]
            tf.config.experimental.set_memory_growth(device, True)
            tf.config.experimental.set_visible_devices(device, "GPU")

    select_gpu_device(gpu_device)

    results = read_results_file(csv_filepath, metrics)

    x_train, y_train, x_test, y_test, y_test_denorm, norm_params = read_data(
        dataset_path, normalization_method, past_history_factor, reduce_data
    )
    print(f"=============== {reduce_data}==============")

        
    print("DATA READED")
    x_train = tf.convert_to_tensor(x_train)
    y_train = tf.convert_to_tensor(y_train)

    x_test = tf.convert_to_tensor(x_test)
    y_test = tf.convert_to_tensor(y_test)
    y_test_denorm = tf.convert_to_tensor(y_test_denorm)
    print("DATA TO TENSOR")
    forecast_horizon = y_test.shape[1]
    past_history = x_test.shape[1]
    steps_per_epoch = min(
        int(np.ceil(x_train.shape[0] / batch_size)), max_steps_per_epoch,
    )
    print("CREATING MODEL..")
    print(model_args)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    print(x_train.shape)
    model = create_model(
        model_name,
        x_train.shape,
        output_size=forecast_horizon,
        optimizer=optimizer,
        loss="mae",
        **model_args
    )

    print("MODEL CREATED")
    print(model.summary())
    print(x_train.shape)
    print(y_train.shape)
    training_time_0 = time.time()

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=(x_test, y_test),
        shuffle=True,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5,  restore_best_weights=True)]
    )
    training_time = time.time() - training_time_0

    # Get validation metrics
    test_time_0 = time.time()

    test_forecast = model(x_test)
    if type(test_forecast) == list:
        test_forecast = test_forecast[-1].numpy()
    else:
        test_forecast = test_forecast.numpy()
    print(test_forecast.shape)
    test_time = time.time() - test_time_0

    for i, nparams in enumerate(norm_params):
        test_forecast[i] = denormalize(
            test_forecast[i], nparams, method=normalization_method,
        )
    print("HOLA")
    if 'group_steps' in model_args.keys() and model_args['group_steps'][-1]>1:
        y_test_denorm = tf.math.reduce_mean(tf.signal.frame(y_test_denorm, model_args['group_factors'][-1], model_args['group_steps'][-1], axis=1), axis=2).numpy()
        
    print(test_forecast.shape)
    print(y_test_denorm.shape)
    if metrics:
        test_metrics = evaluate(y_test_denorm, test_forecast, metrics)
    else:
        test_metrics = {}


    # Save results
    predictions_path = "{}/{}/{}/{}/{}/{}/{}/{}/".format(
        results_path,
        dataset,
        normalization_method,
        past_history_factor,
        epochs,
        batch_size,
        learning_rate,
        model_name
    )
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)
    np.save(
        predictions_path + str(model_index) + str(model_args) + str(reduce_data)  + ".npy", test_forecast,
    )
    np.save(
        predictions_path + str(model_index)+ str(model_args) + str(reduce_data) + ".npy", test_forecast,
    )
    results = results.append(
        {
            "DATASET": dataset,
            "MODEL": model_name,
            "MODEL_INDEX": model_index,
            "MODEL_DESCRIPTION": str(model_args),
            "FORECAST_HORIZON": forecast_horizon,
            "PAST_HISTORY_FACTOR": past_history_factor,
            "PAST_HISTORY": past_history,
            "BATCH_SIZE": batch_size,
            "EPOCHS": epochs,
            "STEPS": steps_per_epoch,
            "OPTIMIZER": "Adam",
            "LEARNING_RATE": learning_rate,
            "NORMALIZATION": normalization_method,
            "TEST_TIME": test_time,
            "TRAINING_TIME": training_time,
            **test_metrics,
            "LOSS": str(history.history["loss"]),
            "VAL_LOSS": str(history.history["val_loss"]),
            "REDUCE_DATA": str(reduce_data) 
        },
        ignore_index=True,
    )

    results.to_csv(
        csv_filepath, sep=";",
    )

    gc.collect()
    del model, x_train, x_test, y_train, y_test, y_test_denorm, test_forecast


def run_experiment(
    error_dict,
    gpu_device,
    dataset,
    dataset_path,
    results_path,
    csv_filepath,
    metrics,
    epochs,
    normalization_method,
    past_history_factor,
    max_steps_per_epoch,
    batch_size,
    learning_rate,
    model_name,
    model_index,
    model_args,
    reduce_data
):
    try:
        _run_experiment(
            gpu_device,
            dataset,
            dataset_path,
            results_path,
            csv_filepath,
            metrics,
            epochs,
            normalization_method,
            past_history_factor,
            max_steps_per_epoch,
            batch_size,
            learning_rate,
            model_name,
            model_index,
            model_args,
            reduce_data
        )
    except Exception as e:
        error_dict["status"] = -1
        error_dict["message"] = str(e)
        print(e)
    else:
        error_dict["status"] = 1


def main(args):
    datasets = args.datasets
    models = args.models
    results_path = args.output
    gpu_device = args.gpu
    metrics = args.metrics
    csv_filename = args.csv_filename
    reduce_data = args.reduce

    parameters = None
    with open(args.parameters, "r") as params_file:
        parameters = json.load(params_file)

    check_params(datasets, models, results_path, parameters, metrics, csv_filename)

    if len(models) == 0:
        models = list(parameters["model_params"].keys())

    if metrics is None:
        metrics = list(METRICS.keys())

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    for dataset_index, dataset_path in enumerate(datasets):
        dataset = os.path.basename(os.path.normpath(dataset_path))

        csv_filepath = results_path + "/{}/{}".format(dataset, csv_filename)
        results = read_results_file(csv_filepath, metrics)
        current_index = results.shape[0]
        print("CURRENT INDEX", current_index)

        experiments_index = 0
        num_total_experiments = np.prod(
            [len(parameters[k]) for k in parameters.keys() if k != "model_params"]
            + [
                np.sum(
                    [
                        np.prod(
                            [
                                len(parameters["model_params"][m][k])
                                for k in parameters["model_params"][m].keys()
                            ]
                        )
                        for m in models
                    ]
                )
            ]
        )

        for epochs, normalization_method, past_history_factor in itertools.product(
            parameters["epochs"],
            parameters["normalization_method"],
            parameters["past_history_factor"],
        ):
            for batch_size, learning_rate in itertools.product(
                parameters["batch_size"], parameters["learning_rate"],
            ):
                for model_name in models:
                    for model_index, model_args in enumerate(
                        product(**parameters["model_params"][model_name])
                    ):
                        
                        experiments_index += 1
                        #if experiments_index <= current_index:
                        #continue

                        # Run each experiment in a new Process to avoid GPU memory leaks
                        manager = Manager()
                        error_dict = manager.dict()
                        run_experiment(error_dict,
                                gpu_device,
                                dataset,
                                dataset_path,
                                results_path,
                                csv_filepath,
                                metrics,
                                epochs,
                                normalization_method,
                                past_history_factor,
                                parameters["max_steps_per_epoch"][0],
                                batch_size,
                                learning_rate,
                                model_name,
                                model_index,
                                model_args,
                                reduce_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        default=[],
        help="Dataset path to experiment over (separated by comma)",
    )
    parser.add_argument(
        "-m",
        "--models",
        nargs="*",
        default=[],
        help="Models to experiment over (separated by comma)",
    )
    parser.add_argument(
        "-p", "--parameters", help="Parameters file path",
    )
    parser.add_argument(
        "-o", "--output", default="./results", help="Output path",
    )
    parser.add_argument(
        "-c", "--csv_filename", default="results.csv", help="Output csv filename",
    )
    parser.add_argument("-g", "--gpu", type=int, default=None, help="GPU device")
    parser.add_argument("-r", "--reduce", type=float, default=None, help="Reduce Data")
    parser.add_argument(
        "-s",
        "--metrics",
        nargs="*",
        default=None,
        help="Metrics to use for evaluation. If not define it will use all possible metrics.",
    )
    args = parser.parse_args()
    
    main(args)
