import pytest
import numpy as np
import pandas as pd
import torch
from tft_torch import tft


def test_tft():
    """ Test the TemporalFusionTransformer module"""
    data_props = {'num_historical_numeric': 4,
                  'num_historical_categorical': 6,
                  'num_static_numeric': 10,
                  'num_static_categorical': 11,
                  'num_future_numeric': 2,
                  'num_future_categorical': 3,
                  'historical_categorical_cardinalities': (1 + np.random.randint(10, size=6)).tolist(),
                  'static_categorical_cardinalities': (1 + np.random.randint(10, size=11)).tolist(),
                  'future_categorical_cardinalities': (1 + np.random.randint(10, size=3)).tolist(),
                  }

    configuration = {
        'model':
            {
                'dropout': 0.05,
                'state_size': 64,
                'output_quantiles': [0.1, 0.5, 0.9],
                'lstm_layers': 2,
                'attention_heads': 4
            },
        # these arguments are related to possible extensions of the model class
        'task_type': 'regression',
        'target_window_start': None,
        'data_props': data_props
    }

    model = tft.TemporalFusionTransformer(configuration)

    # create batch
    batch_size = 256
    historical_steps = 90
    future_steps = 30

    batch = {
        'static_feats_numeric': torch.rand(batch_size, data_props['num_static_numeric'], dtype=torch.float32),
        'static_feats_categorical': torch.stack([
            torch.randint(c, size=(batch_size,)) for c in data_props['static_categorical_cardinalities']
        ], dim=-1).long(),
        'historical_ts_numeric': torch.rand(
            batch_size, historical_steps, data_props['num_historical_numeric'], dtype=torch.float32
        ),
        'historical_ts_categorical': torch.stack([
            torch.randint(c, size=(batch_size, historical_steps))
            for c in data_props['historical_categorical_cardinalities']
        ], dim=-1).long(),
        'future_ts_numeric': torch.rand(
            batch_size, future_steps, data_props['num_future_numeric'], dtype=torch.float32
        ),
        'future_ts_categorical': torch.stack([
            torch.randint(c, size=(batch_size, future_steps))
            for c in data_props['future_categorical_cardinalities']
        ], dim=-1).long(),
    }

    batch_outputs = model(
        batch['historical_ts_numeric'],
        batch['historical_ts_categorical'],
        batch['future_ts_numeric'],
        batch['future_ts_categorical'],
        batch['static_feats_numeric'],
        batch['static_feats_categorical'],
    )

    assert isinstance(batch_outputs, tuple)
    assert len(batch_outputs) == 5

    num_quantiles = len(configuration['model']['output_quantiles'])
    num_static = data_props['num_static_numeric'] + data_props['num_static_categorical']
    num_historical = data_props['num_historical_numeric'] + data_props['num_historical_categorical']
    num_future = data_props['num_future_numeric'] + data_props['num_future_categorical']

    verify_output(batch_outputs=batch_outputs,
                  batch_size=batch_size,
                  future_steps=future_steps,
                  historical_steps=historical_steps,
                  num_future=num_future,
                  num_historical=num_historical,
                  num_quantiles=num_quantiles,
                  num_static=num_static)


def test_tft_invalid_config():
    """
    Verify that exceptions are raised when the provided configuration is invalid in terms of number
    of inputs
    """

    # set the number of historical inputs (both categorical and numeric) to zero
    data_props = {'num_historical_numeric': 0,
                  'num_historical_categorical': 0,
                  'num_static_numeric': 10,
                  'num_static_categorical': 11,
                  'num_future_numeric': 2,
                  'num_future_categorical': 3,
                  'historical_categorical_cardinalities': (1 + np.random.randint(10, size=6)).tolist(),
                  'static_categorical_cardinalities': (1 + np.random.randint(10, size=11)).tolist(),
                  'future_categorical_cardinalities': (1 + np.random.randint(10, size=3)).tolist(),
                  }

    configuration = {
        'model':
            {
                'dropout': 0.05,
                'state_size': 64,
                'output_quantiles': [0.1, 0.5, 0.9],
                'lstm_layers': 2,
                'attention_heads': 4
            },
        # these arguments are related to possible extensions of the model class
        'task_type': 'regression',
        'target_window_start': None,
        'data_props': data_props
    }

    # verify that exception is raised
    with pytest.raises(Exception) as _:
        _ = tft.TemporalFusionTransformer(configuration)

    # set missing data_props
    configuration['data_props'] = {}

    # verify that exception is raised
    with pytest.raises(Exception) as _:
        _ = tft.TemporalFusionTransformer(configuration)


def test_tft_without_categorical_vars():
    """ Test the TemporalFusionTransformer module when no categorical variables are expected"""
    data_props = {'num_historical_numeric': 4,
                  'num_static_numeric': 10,
                  'num_future_numeric': 2,
                  }

    configuration = {
        'model':
            {
                'dropout': 0.05,
                'state_size': 64,
                'output_quantiles': [0.1, 0.5, 0.9],
                'lstm_layers': 2,
                'attention_heads': 4
            },
        # these arguments are related to possible extensions of the model class
        'task_type': 'regression',
        'target_window_start': None,
        'data_props': data_props
    }

    model = tft.TemporalFusionTransformer(configuration)

    # create batch
    batch_size = 256
    historical_steps = 90
    future_steps = 30

    batch = {
        'static_feats_numeric': torch.rand(batch_size, data_props['num_static_numeric'],
                                           dtype=torch.float32),
        'historical_ts_numeric': torch.rand(batch_size, historical_steps, data_props['num_historical_numeric'],
                                            dtype=torch.float32),
        'future_ts_numeric': torch.rand(batch_size, future_steps, data_props['num_future_numeric'],
                                        dtype=torch.float32)
    }

    batch_outputs = model(
        batch['historical_ts_numeric'],
        torch.empty((batch_size, historical_steps, 0)),
        batch['future_ts_numeric'],
        torch.empty((batch_size, future_steps, 0)),
        batch['static_feats_numeric'],
        torch.empty((batch_size, 0)),
    )

    assert isinstance(batch_outputs, tuple)
    assert len(batch_outputs) == 5

    num_quantiles = len(configuration['model']['output_quantiles'])
    num_static = data_props.get('num_static_numeric', 0) + data_props.get('num_static_categorical', 0)
    num_historical = data_props.get('num_historical_numeric', 0) + data_props.get('num_historical_categorical', 0)
    num_future = data_props.get('num_future_numeric', 0) + data_props.get('num_future_categorical', 0)

    verify_output(batch_outputs=batch_outputs,
                  batch_size=batch_size,
                  future_steps=future_steps,
                  historical_steps=historical_steps,
                  num_future=num_future,
                  num_historical=num_historical,
                  num_quantiles=num_quantiles,
                  num_static=num_static)


def test_tft_without_numeric_vars():
    """ Test the TemporalFusionTransformer module when no numeric variables are expected"""
    data_props = {'num_historical_categorical': 6,
                  'num_static_categorical': 11,
                  'num_future_categorical': 3,
                  'historical_categorical_cardinalities': (1 + np.random.randint(10, size=6)).tolist(),
                  'static_categorical_cardinalities': (1 + np.random.randint(10, size=11)).tolist(),
                  'future_categorical_cardinalities': (1 + np.random.randint(10, size=3)).tolist(),
                  }

    configuration = {
        'model':
            {
                'dropout': 0.05,
                'state_size': 64,
                'output_quantiles': [0.1, 0.5, 0.9],
                'lstm_layers': 2,
                'attention_heads': 4
            },
        # these arguments are related to possible extensions of the model class
        'task_type': 'regression',
        'target_window_start': None,
        'data_props': data_props
    }

    model = tft.TemporalFusionTransformer(configuration)

    # create batch
    batch_size = 256
    historical_steps = 90
    future_steps = 30

    batch = {
        'static_feats_categorical': torch.stack([torch.randint(c, size=(batch_size,)) for c in
                                                 data_props['static_categorical_cardinalities']],
                                                dim=-1).type(torch.LongTensor),
        'historical_ts_categorical': torch.stack([torch.randint(c, size=(batch_size, historical_steps)) for c in
                                                  data_props['historical_categorical_cardinalities']],
                                                 dim=-1).type(torch.LongTensor),
        'future_ts_categorical': torch.stack([torch.randint(c, size=(batch_size, future_steps)) for c in
                                              data_props['future_categorical_cardinalities']],
                                             dim=-1).type(torch.LongTensor),

    }

    batch_outputs = model(
        torch.empty((batch_size, historical_steps, 0)),
        batch['historical_ts_categorical'],
        torch.empty((batch_size, future_steps, 0)),
        batch['future_ts_categorical'],
        torch.empty((batch_size, 0)),
        batch['static_feats_categorical'],
    )

    assert isinstance(batch_outputs, tuple)
    assert len(batch_outputs) == 5

    num_quantiles = len(configuration['model']['output_quantiles'])
    num_static = data_props.get('num_static_numeric', 0) + data_props.get('num_static_categorical', 0)
    num_historical = data_props.get('num_historical_numeric', 0) + data_props.get('num_historical_categorical', 0)
    num_future = data_props.get('num_future_numeric', 0) + data_props.get('num_future_categorical', 0)

    verify_output(batch_outputs=batch_outputs,
                  batch_size=batch_size,
                  future_steps=future_steps,
                  historical_steps=historical_steps,
                  num_future=num_future,
                  num_historical=num_historical,
                  num_quantiles=num_quantiles,
                  num_static=num_static)


def verify_output(
    batch_outputs,
    batch_size,
    future_steps,
    historical_steps,
    num_future,
    num_historical,
    num_quantiles,
    num_static,
):
    predicted_quantiles, static_weights, hist_weights, fut_weights, attn = batch_outputs
    assert len(predicted_quantiles.shape) == 3 and all(
        a == b for a, b in zip(predicted_quantiles.shape, [batch_size, future_steps, num_quantiles])
    )
    assert len(static_weights.shape) == 2 and all(
        a == b for a, b in zip(static_weights.shape, [batch_size, num_static])
    )
    assert len(hist_weights.shape) == 3 and all(
        a == b for a, b in zip(hist_weights.shape, [batch_size, historical_steps, num_historical])
    )
    assert len(fut_weights.shape) == 3 and all(
        a == b for a, b in zip(fut_weights.shape, [batch_size, future_steps, num_future])
    )
    assert len(attn.shape) == 3 and all(
        a == b
        for a, b in zip(attn.shape, [batch_size, future_steps, historical_steps + future_steps])
    )
