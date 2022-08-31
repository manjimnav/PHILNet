import tensorflow as tf
from tensorflow_addons.layers import ESN
from tcn import TCN
from functools import partial


def mlp(
        input_shape,
        output_size=1,
        optimizer="adam",
        loss="mae",
        hidden_layers=[32, 16, 8],
        dropout=0.0,
):
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])
    x = tf.keras.layers.Flatten()(inputs)  # Convert the 2d input in a 1d array
    for hidden_units in hidden_layers:
        x = tf.keras.layers.Dense(hidden_units)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss=loss)

    return model


def ernn(
        input_shape,
        output_size=1,
        optimizer="adam",
        loss="mae",
        recurrent_units=[50],
        recurrent_dropout=0,
        return_sequences=False,
        dense_layers=[],
        dense_dropout=0,
):
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])
    # LSTM layers
    return_sequences_tmp = return_sequences if len(recurrent_units) == 1 else True
    x = tf.keras.layers.SimpleRNN(
        recurrent_units[0],
        return_sequences=return_sequences_tmp,
        dropout=recurrent_dropout,
    )(inputs)
    for i, u in enumerate(recurrent_units[1:]):
        return_sequences_tmp = (
            return_sequences if i == len(recurrent_units) - 2 else True
        )
        x = tf.keras.layers.SimpleRNN(
            u, return_sequences=return_sequences_tmp, dropout=recurrent_dropout
        )(x)
    # Dense layers
    if return_sequences:
        x = tf.keras.layers.Flatten()(x)
    for hidden_units in dense_layers:
        x = tf.keras.layers.Dense(hidden_units)(x)
        if dense_dropout > 0:
            x = tf.keras.layers.Dropout(dense_dropout)(dense_dropout)
    x = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss=loss)

    return model


def esn(
        input_shape,
        output_size=1,
        optimizer="adam",
        loss="mae",
        recurrent_units=[50],
        return_sequences=False,
        dense_layers=[32],
        dense_dropout=0,
):
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])
    # LSTM layers
    return_sequences_tmp = return_sequences if len(recurrent_units) == 1 else True
    x = ESN(recurrent_units[0], return_sequences=return_sequences_tmp, use_norm2=True)(
        inputs
    )
    for i, u in enumerate(recurrent_units[1:]):
        return_sequences_tmp = (
            return_sequences if i == len(recurrent_units) - 2 else True
        )
        x = ESN(u, return_sequences=return_sequences_tmp, use_norm2=True)(x)
    # Dense layers
    if return_sequences:
        x = tf.keras.layers.Flatten()(x)
    for hidden_units in dense_layers:
        x = tf.keras.layers.Dense(hidden_units)(x)
        if dense_dropout > 0:
            x = tf.keras.layers.Dropout(dense_dropout)(dense_dropout)
    x = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss=loss)

    return model


def lstm(
        input_shape,
        output_size=1,
        optimizer="adam",
        loss="mae",
        recurrent_units=[50],
        recurrent_dropout=0,
        return_sequences=False,
        dense_layers=[],
        dense_dropout=0,
):
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])
    # LSTM layers
    return_sequences_tmp = return_sequences if len(recurrent_units) == 1 else True
    x = tf.keras.layers.LSTM(
        recurrent_units[0],
        return_sequences=return_sequences_tmp,
        dropout=recurrent_dropout,
    )(inputs)
    for i, u in enumerate(recurrent_units[1:]):
        return_sequences_tmp = (
            return_sequences if i == len(recurrent_units) - 2 else True
        )
        x = tf.keras.layers.LSTM(
            u, return_sequences=return_sequences_tmp, dropout=recurrent_dropout
        )(x)
    # Dense layers
    if return_sequences:
        x = tf.keras.layers.Flatten()(x)
    for hidden_units in dense_layers:
        x = tf.keras.layers.Dense(hidden_units)(x)
        if dense_dropout > 0:
            x = tf.keras.layers.Dropout(dense_dropout)(dense_dropout)
    x = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss=loss)

    return model


class hierarchical_loss(tf.keras.losses.Loss):

    def __init__(self, base_criterion, gf, ge,  reduction=tf.keras.losses.Reduction.AUTO):
        self.gf = gf
        self.ge = ge
        self.base_criterion = base_criterion
        self.reduction = reduction

    def __call__(self, y_true, y_pred, sample_weight=None):
        criterion = tf.keras.losses.get(self.base_criterion)

        y_h_true = tf.math.reduce_mean(tf.signal.frame(y_true, self.gf, self.ge, axis=1), axis=2)

        loss = criterion(y_pred, tf.squeeze(y_h_true))

        return loss


def hlnet(
        input_shape,
        output_size=1,
        optimizer="adam",
        loss="mae",
        recurrent_units=[50],
        recurrent_dropout=0,
        return_sequences=False,
        dense_layers=[],
        dense_dropout=0,
        group_factors=[1],
        group_steps=[1],
        reduce_imputs=True
):
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])
    # LSTM layers
    outputs = []
    prev_hidden = None
    for gf, ge in zip(group_factors, group_steps):
        return_sequences_tmp = return_sequences if len(recurrent_units) == 1 else True
        print(f"=============={reduce_imputs}================")
        if reduce_imputs:
            inputs_gr = tf.keras.layers.Lambda(
                lambda inp: tf.math.reduce_mean(tf.signal.frame(inp, gf, ge, axis=1), axis=2))(inputs)
        else:
            inputs_gr = inputs
            
        x = tf.keras.layers.LSTM(
            recurrent_units[0],
            return_sequences=return_sequences_tmp,
            dropout=recurrent_dropout,
        )(inputs_gr)

        for i, u in enumerate(recurrent_units[1:]):
            return_sequences_tmp = (
                return_sequences if i == len(recurrent_units) - 2 else True
            )
            x = tf.keras.layers.LSTM(
                u, return_sequences=return_sequences_tmp, dropout=recurrent_dropout
            )(x)

        # Dense layers
        if return_sequences:

            x = tf.keras.layers.Flatten()(x)

        if prev_hidden is not None:
            x = tf.keras.layers.Concatenate(axis=1)([x, prev_hidden])
        else:
            x_hidden = tf.identity(x)
            prev_hidden = tf.stop_gradient(x_hidden)

        for hidden_units in dense_layers:
            x = tf.keras.layers.Dense(hidden_units//ge)(x)
            if dense_dropout > 0:
                x = tf.keras.layers.Dropout(dense_dropout)(dense_dropout)

        layer_out = tf.keras.layers.Dense(output_size- gf +1, name=f'level_{gf}_out')(x)
        outputs.append(layer_out)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    losses = {
        f"level_{gf}_out": hierarchical_loss(base_criterion=loss, gf=gf, ge=ge, reduction=tf.keras.losses.Reduction.SUM)
        for i, (gf, ge) in enumerate(zip(group_factors, group_steps))
    }
    model.compile(optimizer=optimizer, loss=losses)

    return model

def philnet(
        input_shape,
        output_size=1,
        optimizer="adam",
        loss="mae",
        recurrent_units=[50],
        recurrent_dropout=0,
        return_sequences=False,
        dense_layers=[],
        dense_dropout=0,
        group_factors=[1],
        group_steps=[1],
        reduce_imputs=True
):
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])
    # LSTM layers
    outputs = []
    prev_hidden = None
    for gf, ge in zip(group_factors, group_steps):
        return_sequences_tmp = return_sequences if len(recurrent_units) == 1 else True
        print(f"=============={reduce_imputs}================")
        if reduce_imputs:
            inputs_gr = tf.keras.layers.Lambda(
                lambda inp: tf.math.reduce_mean(tf.signal.frame(inp, gf, ge, axis=1), axis=2))(inputs)
        else:
            inputs_gr = inputs
            
        x = tf.keras.layers.LSTM(
            recurrent_units[0],
            return_sequences=return_sequences_tmp,
            dropout=recurrent_dropout,
        )(inputs_gr)

        for i, u in enumerate(recurrent_units[1:]):
            return_sequences_tmp = (
                return_sequences if i == len(recurrent_units) - 2 else True
            )
            x = tf.keras.layers.LSTM(
                u, return_sequences=return_sequences_tmp, dropout=recurrent_dropout
            )(x)

        # Dense layers
        if return_sequences:

            x = tf.keras.layers.Flatten()(x)

        if prev_hidden is not None:
            x = tf.keras.layers.Concatenate(axis=1)([x, prev_hidden])
        else:
            x_hidden = tf.identity(x)
            prev_hidden = x_hidden

        for hidden_units in dense_layers:
            x = tf.keras.layers.Dense(hidden_units)(x)
            if dense_dropout > 0:
                x = tf.keras.layers.Dropout(dense_dropout)(dense_dropout)

        layer_out = tf.keras.layers.Dense(output_size - gf +1, name=f'level_{gf}_out')(x)
        outputs.append(layer_out)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    losses = {
        f"level_{gf}_out": hierarchical_loss(base_criterion=loss, gf=gf, ge=ge, reduction=tf.keras.losses.Reduction.SUM)
        for i, (gf, ge) in enumerate(zip(group_factors, group_steps))
    }
    model.compile(optimizer=optimizer, loss=losses)

    return model

def hlcnn(
        input_shape,
        output_size=1,
        optimizer="adam",
        loss="mae",
	    conv_layers=[64, 128],
        kernel_sizes=[7, 5],
        pool_sizes=[2, 2],
        dense_layers=[],
        dense_dropout=0.0,
        group_factors=[1],
        group_steps=[1],
        reduce_inputs=True
):
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])
    # LSTM layers
    outputs = []
    prev_hidden = None
    for gf, ge in zip(group_factors, group_steps):
        if reduce_inputs:
            inputs_gr = tf.keras.layers.Lambda(
                lambda inp: tf.math.reduce_sum(tf.signal.frame(inp, gf, ge, axis=1), axis=2))(inputs)
        else:
            inputs_gr = inputs
        x = tf.keras.layers.Conv1D(
        conv_layers[0], kernel_sizes[0], activation="relu", padding="same")(inputs_gr)

        for chanels, kernel_size, pool_size in zip(
            conv_layers[1:], kernel_sizes[1:], pool_sizes[1:]):
            x = tf.keras.layers.Conv1D(
                chanels, kernel_size, activation="relu", padding="same"
            )(x)

        x = tf.keras.layers.Concatenate(axis=-1)([x, inputs_gr])
        # Dense layer
        x = tf.keras.layers.Flatten()(x)

        if prev_hidden is not None:
            x = tf.keras.layers.Concatenate(axis=1)([x, prev_hidden])
        else:
            x_hidden = tf.identity(x)
            prev_hidden = x_hidden
        for hidden_units in dense_layers:
            x = tf.keras.layers.Dense(hidden_units)(x)

        layer_out = tf.keras.layers.Dense(output_size - gf + 1, name=f'level_{gf}_out')(x)

        outputs.append(layer_out)
        
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    print(model)
    losses = {
        f"level_{gf}_out": hierarchical_loss(base_criterion=loss, gf=gf, ge=ge, reduction=tf.keras.losses.Reduction.SUM)
        for i, (gf, ge) in enumerate(zip(group_factors, group_steps))
    }
    model.compile(optimizer=optimizer, loss=losses)
    print(model)
    return model

def gru(
        input_shape,
        output_size=1,
        optimizer="adam",
        loss="mae",
        recurrent_units=[50],
        recurrent_dropout=0,
        return_sequences=False,
        dense_layers=[],
        dense_dropout=0,
):
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])
    # LSTM layers
    return_sequences_tmp = return_sequences if len(recurrent_units) == 1 else True
    x = tf.keras.layers.GRU(
        recurrent_units[0],
        return_sequences=return_sequences_tmp,
        dropout=recurrent_dropout,
    )(inputs)
    for i, u in enumerate(recurrent_units[1:]):
        return_sequences_tmp = (
            return_sequences if i == len(recurrent_units) - 2 else True
        )
        x = tf.keras.layers.GRU(
            u, return_sequences=return_sequences_tmp, dropout=recurrent_dropout
        )(x)
    # Dense layers
    if return_sequences:
        x = tf.keras.layers.Flatten()(x)
    for hidden_units in dense_layers:
        x = tf.keras.layers.Dense(hidden_units)(x)
        if dense_dropout > 0:
            x = tf.keras.layers.Dropout(dense_dropout)(dense_dropout)
    x = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss=loss)

    return model


def cnn(
        input_shape,
        output_size=1,
        optimizer="adam",
        loss="mae",
        conv_layers=[64, 128],
        kernel_sizes=[7, 5],
        pool_sizes=[2, 2],
        dense_layers=[],
        dense_dropout=0.0,
):
    assert len(conv_layers) == len(kernel_sizes)
    assert 0 <= dense_dropout <= 1
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])
    # First conv block
    x = tf.keras.layers.Conv1D(
        conv_layers[0], kernel_sizes[0], activation="relu", padding="same"
    )(inputs)
    x = tf.keras.layers.Conv1D(
        conv_layers[0], kernel_sizes[0], activation="relu", padding="same"
    )(x)
    if pool_sizes[0] and x.shape[-2] // pool_sizes[0] > 1:
        x = tf.keras.layers.MaxPool1D(pool_size=pool_sizes[0])(x)
    # Rest of the conv blocks
    for chanels, kernel_size, pool_size in zip(
            conv_layers[1:], kernel_sizes[1:], pool_sizes[1:]):
        x = tf.keras.layers.Conv1D(
            chanels, kernel_size, activation="relu", padding="same"
        )(x)
        x = tf.keras.layers.Conv1D(
            chanels, kernel_size, activation="relu", padding="same"
        )(x)
        if pool_size and x.shape[-2] // pool_size > 1:
            x = tf.keras.layers.MaxPool1D(pool_size=pool_size)(x)
    # Dense block
    x = tf.keras.layers.Flatten()(x)
    for hidden_units in dense_layers:
        x = tf.keras.layers.Dense(hidden_units)(x)
        if dense_dropout > 0:
            tf.keras.layers.Dropout(dense_dropout)(x)
    x = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def tcn(
        input_shape,
        output_size=1,
        optimizer="adam",
        loss="mae",
        nb_filters=64,
        kernel_size=2,
        nb_stacks=1,
        dilations=[1, 2, 4, 8, 16],
        tcn_dropout=0.0,
        return_sequences=True,
        activation="linear",
        padding="causal",
        use_skip_connections=True,
        use_batch_norm=False,
        dense_layers=[],
        dense_dropout=0.0,
):
    inputs = tf.keras.layers.Input(shape=input_shape[-2:])

    x = TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        nb_stacks=nb_stacks,
        dilations=dilations,
        use_skip_connections=use_skip_connections,
        dropout_rate=tcn_dropout,
        activation=activation,
        use_batch_norm=use_batch_norm,
        padding=padding,
    )(inputs)
    # Dense block
    if return_sequences:
        x = tf.keras.layers.Flatten()(x)
    for hidden_units in dense_layers:
        x = tf.keras.layers.Dense(hidden_units)(x)
        if dense_dropout > 0:
            tf.keras.layers.Dropout(dense_dropout)(x)
    x = tf.keras.layers.Dense(output_size)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss=loss)

    return model


def create_rnn(func):
    return lambda input_shape, output_size, optimizer, loss, **args: func(
        input_shape=input_shape,
        output_size=output_size,
        optimizer=optimizer,
        loss="mae",
        recurrent_units=[args["units"]] * args["layers"],
        return_sequences=args["return_sequence"],
    )


def create_hrnn(func):
    return lambda input_shape, output_size, optimizer, loss, **args: func(
        input_shape=input_shape,
        output_size=output_size,
        optimizer=optimizer,
        loss="mae",
        recurrent_units=[args["units"]] * args["layers"],
        return_sequences=args["return_sequence"],
        group_factors=args['group_factors'],
        group_steps=args['group_steps']
    )


def create_cnn(func):
    return lambda input_shape, output_size, optimizer, loss, **args: func(input_shape,
        output_size=output_size,
        optimizer=optimizer,
        loss=loss,
        conv_layers=[b[0] for b in args['conv_blocks']],
        kernel_sizes=[b[1] for b in args['conv_blocks']],
        pool_sizes=[b[2] for b in args['conv_blocks']]
    )
    
def create_hcnn(func):
    return lambda input_shape, output_size, optimizer, loss, **args: func(input_shape,
        output_size=output_size,
        optimizer=optimizer,
        loss=loss,
        conv_layers=[b[0] for b in args['conv_blocks']],
        kernel_sizes=[b[1] for b in args['conv_blocks']],
        pool_sizes=[b[2] for b in args['conv_blocks']],
        group_factors=args['group_factors'],
        group_steps=args['group_steps']
    )



model_factory = {
    "mlp": mlp,
    "ernn": create_rnn(ernn),
    "lstm": create_rnn(lstm),
    "gru": create_rnn(gru),
    "esn": create_rnn(esn),
    "cnn": create_cnn(cnn),
    "tcn": tcn,
    "hlnet": create_hrnn(hlnet),
    "philnet": create_hrnn(philnet),
    "hlcnn": create_hcnn(hlcnn)
}


def create_model(model_name, input_shape, **args):
    assert model_name in model_factory.keys(), "Model '{}' not supported".format(
        model_name
    )
    return model_factory[model_name](input_shape, **args)

    

