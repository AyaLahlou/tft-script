import copy
import math
from typing import List, Dict, Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F
from tft_torch.base_blocks import TimeDistributed, NullTransform
from typing import Dict, Tuple, List, Optional
from torch.nn import Module


class GatedLinearUnit(nn.Module):
    """
    This module is also known as  **GLU** - Formulated in:
    `Dauphin, Yann N., et al. "Language modeling with gated convolutional networks."
    International conference on machine learning. PMLR, 2017
    <https://arxiv.org/abs/1612.08083>`_.

    The output of the layer is a linear projection (X * W + b) modulated by the gates **sigmoid** (X * V + c).
    These gates multiply each element of the matrix X * W + b and control the information passed on in the hierarchy.
    This unit is a simplified gating mechanism for non-deterministic gates that reduce the vanishing gradient problem,
    by having linear units coupled to the gates. This retains the non-linear capabilities of the layer while allowing
    the gradient to propagate through the linear unit without scaling.

    Parameters
    ----------
    input_dim: int
        The embedding size of the input.
    """

    def __init__(self, input_dim: int):
        super(GatedLinearUnit, self).__init__()

        # Two dimension-preserving dense layers
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)


class GatedResidualNetwork(nn.Module):
    """
    This module, known as **GRN**, takes in a primary input (x) and an optional context vector (c).
    It uses a ``GatedLinearUnit`` for controlling the extent to which the module will contribute to the original input
    (x), potentially skipping over the layer entirely as the GLU outputs could be all close to zero, by that suppressing
    the non-linear contribution.
    In cases where no context vector is used, the GRN simply treats the context input as zero.
    During training, dropout is applied before the gating layer.

    Parameters
    ----------
    input_dim: int
        The embedding width/dimension of the input.
    hidden_dim: int
        The intermediate embedding width.
    output_dim: int
        The embedding width of the output tensors.
    dropout: Optional[float]
        The dropout rate associated with the component.
    context_dim: Optional[int]
        The embedding width of the context signal expected to be fed as an auxiliary input to this component.
    batch_first: Optional[bool]
        A boolean indicating whether the batch dimension is expected to be the first dimension of the input or not.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.05,
        context_dim: Optional[int] = None,
        batch_first: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        self.project_residual = (input_dim != output_dim)

        # 1) Primary projection + skip
        self.fc1 = TimeDistributed(nn.Linear(input_dim, hidden_dim), batch_first=batch_first)
        if self.project_residual:
            self.skip_layer = TimeDistributed(
                nn.Linear(input_dim, output_dim, bias=False), batch_first=batch_first
            )
        else:
            self.skip_layer = TimeDistributed(NullTransform(), batch_first=batch_first)

        # 2) Nonlinearity + second projection
        self.elu1 = nn.ELU()
        self.fc2  = TimeDistributed(nn.Linear(hidden_dim, output_dim), batch_first=batch_first)

        # 3) Dropout + gating
        self.dropout   = nn.Dropout(dropout)
        # only one gating module – we’ll use this
        self.gate_layer = TimeDistributed(
            nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.Sigmoid()
            ),
            batch_first=batch_first,
        )

        # 4) LayerNorm (name matches forward)
        self.layer_norm = TimeDistributed(nn.LayerNorm(output_dim), batch_first=batch_first)

        # 5) Context projection (always defined)
        if context_dim is not None:
            self.context_projection = TimeDistributed(
                nn.Linear(context_dim, hidden_dim, bias=False),
                batch_first=batch_first
            )
        else:
            self.context_projection = TimeDistributed(NullTransform(), batch_first=batch_first)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # residual path
        residual = self.skip_layer(x)

        # primary path
        x = self.fc1(x)
        if self.context_dim is not None and context is not None:
            x = x + self.context_projection(context)

        # second projection + nonlinearity + dropout + gating
        x = self.fc2(x)
        x = self.elu1(x)
        x = self.dropout(x)
        gate = self.gate_layer(x)
        x = x * gate

        # add residual + normalize
        x = x + residual
        return self.layer_norm(x)


class VariableSelectionNetwork(nn.Module):
    """
    This module is designed to handle the fact that the relevant and specific contribution of each input variable
    to the  output is typically unknown. This module enables instance-wise variable selection, and is applied to
    both the static covariates and time-dependent covariates.

    Beyond providing insights into which variables are the most significant oones for the prediction problem,
    variable selection also allows the model to remove any unnecessary noisy inputs which could negatively impact
    performance.

    Parameters
    ----------
    input_dim: int
        The attribute/embedding dimension of the input, associated with the ``state_size`` of th model.
    num_inputs: int
        The quantity of input variables, including both numeric and categorical inputs for the relevant channel.
    hidden_dim: int
        The embedding width of the output.
    dropout: float
        The dropout rate associated with ``GatedResidualNetwork`` objects composing this object.
    context_dim: Optional[int]
        The embedding width of the context signal expected to be fed as an auxiliary input to this component.
    batch_first: Optional[bool]
        A boolean indicating whether the batch dimension is expected to be the first dimension of the input or not.
    """

    def __init__(self, input_dim: int, num_inputs: int, hidden_dim: int, dropout: float,
                 context_dim: Optional[int] = None,
                 batch_first: Optional[bool] = True):
        super(VariableSelectionNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_inputs = num_inputs
        self.dropout = dropout
        self.context_dim = context_dim

        # A GRN to apply on the flat concatenation of the input representation (all inputs together),
        # possibly provided with context information
        self.flattened_grn = GatedResidualNetwork(input_dim=self.num_inputs * self.input_dim,
                                                  hidden_dim=self.hidden_dim,
                                                  output_dim=self.num_inputs,
                                                  dropout=self.dropout,
                                                  context_dim=self.context_dim,
                                                  batch_first=batch_first)
        # activation for transforming the GRN output to weights
        self.softmax = nn.Softmax(dim=1)

        # In addition, each input variable (after transformed to its wide representation) goes through its own GRN.
        self.single_variable_grns = nn.ModuleList()
        for _ in range(self.num_inputs):
            self.single_variable_grns.append(
                GatedResidualNetwork(input_dim=self.input_dim,
                                     hidden_dim=self.hidden_dim,
                                     output_dim=self.hidden_dim,
                                     dropout=self.dropout,
                                     batch_first=batch_first))
    def forward(
        self,
        flattened_embedding: torch.Tensor,
        context: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ===========================================================================
        # Infer variable selection weights – using the flattened representation GRN
        # ===========================================================================
        # `flattened_embedding` shape: (num_samples * num_temporal_steps, num_inputs * input_dim)
        sparse_weights = self.flattened_grn(flattened_embedding, context)
        sparse_weights = self.softmax(sparse_weights).unsqueeze(2)
        # sparse_weights shape: (num_samples * num_temporal_steps, num_inputs, 1)

        # ===========================================================================
        # Apply a dedicated GRN to each variable embedding
        # ===========================================================================
        processed_inputs: List[torch.Tensor] = []
        for idx, grn in enumerate(self.single_variable_grns):
            start = idx * self.input_dim
            end   = (idx + 1) * self.input_dim
            # slice out this variable’s features
            inp = flattened_embedding[..., start:end]  # (batch*time, input_dim)
            # transform via its GRN
            processed_inputs.append(grn(inp, context))
        # processed_inputs: list of Tensors each (batch*time, hidden_dim)

        # stack into shape (batch*time, hidden_dim, num_inputs)
        processed_inputs = torch.stack(processed_inputs, dim=-1)

        # ===========================================================================
        # Weight & sum to get final representation
        # ===========================================================================
        # sparse_weights: (batch*time, num_inputs, 1) -> transpose -> (batch*time, 1, num_inputs)
        weights_t = sparse_weights.transpose(1, 2)
        # elementwise multiply: (batch*time, hidden_dim, num_inputs)
        weighted = processed_inputs * weights_t
        # sum over the inputs axis -> (batch*time, hidden_dim)
        outputs = weighted.sum(dim=-1)

        return outputs, sparse_weights


class InputChannelEmbedding(nn.Module):
    """
    A module to handle the transformation/embedding of an input channel composed of numeric tensors and categorical
    tensors.
    It holds a NumericInputTransformation module for handling the embedding of the numeric inputs,
    and a CategoricalInputTransformation module for handling the embedding of the categorical inputs.

    Parameters
    ----------
    state_size : int
        The state size of the model, which determines the embedding dimension/width of each input variable.
    num_numeric : int
        The quantity of numeric input variables associated with the input channel.
    num_categorical : int
        The quantity of categorical input variables associated with the input channel.
    categorical_cardinalities: List[int]
        The quantity of categories associated with each of the categorical input variables.
    time_distribute: Optional[bool]
        A boolean indicating whether to wrap the composing transformations using the ``TimeDistributed`` module.
    """

    def __init__(self, state_size: int, num_numeric: int, num_categorical: int, categorical_cardinalities: List[int],
                 time_distribute: Optional[bool] = False):
        super(InputChannelEmbedding, self).__init__()

        self.state_size = state_size
        self.num_numeric = num_numeric
        self.num_categorical = num_categorical
        self.categorical_cardinalities = categorical_cardinalities
        self.time_distribute = time_distribute

        if (num_numeric + num_categorical) < 1:
            raise ValueError(f"""At least a single input variable (either numeric or categorical) should be included
            as part of the input channel.
            According to the provided configuration:
            num_numeric + num_categorical = {num_numeric} + {num_categorical} = {num_numeric + num_categorical} < 1
            """)

        if self.time_distribute:
            self.numeric_transform = TimeDistributed(
                NumericInputTransformation(num_inputs=num_numeric, state_size=state_size))
            self.categorical_transform = TimeDistributed(
                CategoricalInputTransformation(num_inputs=num_categorical, state_size=state_size,
                                               cardinalities=categorical_cardinalities))
        else:
            self.numeric_transform = NumericInputTransformation(num_inputs=num_numeric, state_size=state_size)
            self.categorical_transform = CategoricalInputTransformation(num_inputs=num_categorical,
                                                                        state_size=state_size,
                                                                        cardinalities=categorical_cardinalities)

        # in case some input types are not expected there is no need in the type specific transformation.
        # instead the "transformation" will return an empty list
        if num_numeric == 0:
            self.numeric_transform = NullTransform()
        if num_categorical == 0:
            self.categorical_transform = NullTransform()

    def forward(self, x_numeric, x_categorical) -> torch.Tensor:
        if x_numeric.numel() > 0:
            batch_size, seq_len = x_numeric.shape[0], x_numeric.shape[1] if len(x_numeric.shape) > 1 else 1
        else:
            batch_size, seq_len = x_categorical.shape[0], x_categorical.shape[1] if len(x_categorical.shape) > 1 else 1

        processed_numeric = self.numeric_transform(x_numeric)
        processed_categorical = self.categorical_transform(x_categorical)

        # Ensure processed_numeric and processed_categorical are always lists
        if not isinstance(processed_numeric, list):
            processed_numeric = [processed_numeric] if processed_numeric.numel() > 0 else []
        if not isinstance(processed_categorical, list):
            processed_categorical = [processed_categorical] if processed_categorical.numel() > 0 else []

        # Combine the processed features
        if len(processed_numeric) > 0 and len(processed_categorical) > 0:
            all_processed = torch.cat(processed_numeric + processed_categorical, dim=1)
        elif len(processed_numeric) > 0:
            all_processed = torch.cat(processed_numeric, dim=1)
        elif len(processed_categorical) > 0:
            all_processed = torch.cat(processed_categorical, dim=1)
        else:
            # Create empty tensor with appropriate dimensions
            if self.time_distribute:
                all_processed = torch.empty(batch_size, seq_len, 0, device=x_numeric.device if x_numeric.numel() > 0 else x_categorical.device)
            else:
                all_processed = torch.empty(batch_size, 0, device=x_numeric.device if x_numeric.numel() > 0 else x_categorical.device)
        
        # For temporal data, reshape to 3D tensor
        if self.time_distribute and all_processed.size(1) > 0:
            merged_transformations = all_processed.view(batch_size, seq_len, -1)
        else:
            merged_transformations = all_processed
        
        return merged_transformations
        # Both of the returned values, "processed_numeric" and "processed_categorical" are lists,
        # with "num_numeric" elements and "num_categorical" respectively - each element in these lists corresponds
        # to a single input variable, and is represent by its embedding, shaped as:
        # [(num_samples * num_temporal_steps) x state_size]
        # (for the static input channel, num_temporal_steps is irrelevant and can be treated as 1



class NumericInputTransformation(nn.Module):
    """
    A module for transforming/embeddings the set of numeric input variables from a single input channel.
    Each input variable will be projected using a dedicated linear layer to a vector with width state_size.
    The result of applying this module is a list, with length num_inputs, that contains the embedding of each input
    variable for all the observations and time steps.

    Parameters
    ----------
    num_inputs : int
        The quantity of numeric input variables associated with this module.
    state_size : int
        The state size of the model, which determines the embedding dimension/width.
    """

    def __init__(self, num_inputs: int, state_size: int):
        super(NumericInputTransformation, self).__init__()
        self.num_inputs = num_inputs
        self.state_size = state_size

        self.numeric_projection_layers = nn.ModuleList()
        for _ in range(self.num_inputs):
            self.numeric_projection_layers.append(nn.Linear(1, self.state_size))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # every input variable is projected using its dedicated linear layer,
        # the results are stored as a list
        projections = []
        for i, layer in enumerate(self.numeric_projection_layers):
            projections.append(layer(x[:, [i]]))

        return projections


class CategoricalInputTransformation(nn.Module):
    """
    A module for transforming/embeddings the set of categorical input variables from a single input channel.
    Each input variable will be projected using a dedicated embedding layer to a vector with width state_size.
    The result of applying this module is a list, with length num_inputs, that contains the embedding of each input
    variable for all the observations and time steps.

    Parameters
    ----------
    num_inputs : int
        The quantity of categorical input variables associated with this module.
    state_size : int
        The state size of the model, which determines the embedding dimension/width.
    cardinalities: List[int]
        The quantity of categories associated with each of the input variables.
    """

    def __init__(self, num_inputs: int, state_size: int, cardinalities: List[int]):
        super(CategoricalInputTransformation, self).__init__()
        self.num_inputs = num_inputs
        self.state_size = state_size
        self.cardinalities = cardinalities

        # layers for processing the categorical inputs
        self.categorical_embedding_layers = nn.ModuleList()
        for idx, cardinality in enumerate(self.cardinalities):
            self.categorical_embedding_layers.append(nn.Embedding(cardinality, self.state_size))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # every input variable is projected using its dedicated embedding layer,
        # the results are stored as a list
        embeddings = []
        for i, layer in enumerate(self.categorical_embedding_layers):
            embeddings.append(layer(x[:, i]))

        return embeddings


class GateAddNorm(nn.Module):
    """
    A composite of:
      1) Dropout
      2) GatedLinearUnit
      3) Residual add
      4) LayerNorm
    TorchScript–compatible.
    """

    def __init__(self, input_dim: int, dropout: Optional[float] = None):
        super().__init__()
        # always a float
        self.dropout_rate = dropout if dropout is not None else 0.0
        # always defined (if rate=0, this is a no-op)
        self.dropout_layer = nn.Dropout(self.dropout_rate)

        # gating step
        self.gate = TimeDistributed(GatedLinearUnit(input_dim), batch_first=True)
        # normalization step
        self.layernorm = TimeDistributed(nn.LayerNorm(input_dim), batch_first=True)

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # only apply dropout if rate > 0
        if self.dropout_rate > 0.0:
            x = self.dropout_layer(x)

        # gating
        x = self.gate(x)

        # residual connection if provided
        if residual is not None:
            x = x + residual

        # final normalization
        return self.layernorm(x)


class InterpretableMultiHeadAttention(nn.Module):
    """
    The mechanism implemented in this module is used to learn long-term relationships across different time-steps.
    It is a modified version of multi-head attention, for enhancing explainability. On this modification,
    as opposed to traditional versions of multi-head attention, the "values" signal is shared for all the heads -
    and additive aggregation is employed across all the heads.
    According to the paper, each head can learn different temporal patterns, while attending to a common set of
    input features which can be interpreted as  a simple ensemble over attention weights into a combined matrix, which,
    compared to the original multi-head attention matrix, yields an increased representation capacity in an efficient
    way.

    Parameters
    ----------
    embed_dim: int
        The dimensions associated with the ``state_size`` of th model, corresponding to the input as well as the output.
    num_heads: int
        The number of attention heads composing the Multi-head attention component.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super(InterpretableMultiHeadAttention, self).__init__()

        self.d_model = embed_dim  # the state_size (model_size) corresponding to the input and output dimension
        self.num_heads = num_heads  # the number of attention heads
        self.all_heads_dim = embed_dim * num_heads  # the width of the projection for the keys and queries

        self.w_q = nn.Linear(embed_dim, self.all_heads_dim)  # multi-head projection for the queries
        self.w_k = nn.Linear(embed_dim, self.all_heads_dim)  # multi-head projection for the keys
        self.w_v = nn.Linear(embed_dim, embed_dim)  # a single, shared, projection for the values

        # the last layer is used for final linear mapping (corresponds to W_H in the paper)
        self.out = nn.Linear(self.d_model, self.d_model)

    def forward(self, q, k, v, mask=None):
        num_samples = q.size(0)

        # Dimensions:
        # queries tensor - q: [num_samples x num_future_steps x state_size]
        # keys tensor - k: [num_samples x (num_total_steps) x state_size]
        # values tensor - v: [num_samples x (num_total_steps) x state_size]

        # perform linear operation and split into h heads
        q_proj = self.w_q(q).view(num_samples, -1, self.num_heads, self.d_model)
        k_proj = self.w_k(k).view(num_samples, -1, self.num_heads, self.d_model)
        v_proj = self.w_v(v).repeat(1, 1, self.num_heads).view(num_samples, -1, self.num_heads, self.d_model)

        # transpose to get the following shapes
        q_proj = q_proj.transpose(1, 2)  # (num_samples x num_future_steps x num_heads x state_size)
        k_proj = k_proj.transpose(1, 2)  # (num_samples x num_total_steps x num_heads x state_size)
        v_proj = v_proj.transpose(1, 2)  # (num_samples x num_total_steps x num_heads x state_size)

        # calculate attention using function we will define next
        attn_outputs_all_heads, attn_scores_all_heads = self.attention(q_proj, k_proj, v_proj, mask)
        # Dimensions:
        # attn_scores_all_heads: [num_samples x num_heads x num_future_steps x num_total_steps]
        # attn_outputs_all_heads: [num_samples x num_heads x num_future_steps x state_size]

        # take average along heads
        attention_scores = attn_scores_all_heads.mean(dim=1)
        attention_outputs = attn_outputs_all_heads.mean(dim=1)
        # Dimensions:
        # attention_scores: [num_samples x num_future_steps x num_total_steps]
        # attention_outputs: [num_samples x num_future_steps x state_size]

        # weigh attention outputs
        output = self.out(attention_outputs)
        # output: [num_samples x num_future_steps x state_size]

        return output, attention_outputs, attention_scores

    def attention(self, q, k, v, mask=None):
        # Applying the scaled dot product
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        # Dimensions:
        # attention_scores: [num_samples x num_heads x num_future_steps x num_total_steps]

        # Decoder masking is applied to the multi-head attention layer to ensure that each temporal dimension can only
        # attend to features preceding it
        if mask is not None:
            # the mask is broadcast along the batch(dim=0) and heads(dim=1) dimensions,
            # where the mask==True, the scores are "cancelled" by setting a very small value
            attention_scores = attention_scores.masked_fill(mask, -1e9)

        # still part of the scaled dot-product attention (dimensions are kept)
        attention_scores = F.softmax(attention_scores, dim=-1)
        # matrix multiplication is performed on the last two-dimensions to retrieve attention outputs
        attention_outputs = torch.matmul(attention_scores, v)
        # Dimensions:
        # attention_outputs: [num_samples x num_heads x num_future_steps x state_size]

        return attention_outputs, attention_scores


class TemporalFusionTransformer(nn.Module):
    """
    This class implements the Temporal Fusion Transformer model described in the paper
    `Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
    <https://arxiv.org/abs/1912.09363>`_.

    Parameters
    ----------
    config: DictConfig
        A mapping describing both the expected structure of the input of the model, and the architectural specification
        of the model.
        This mapping should include a key named ``data_props`` in which the dimensions and cardinalities (where the
        inputs are categorical) are specified. Moreover, the configuration mapping should contain a key named ``model``,
        specifying ``attention_heads`` , ``dropout`` , ``lstm_layers`` , ``output_quantiles`` and ``state_size`` ,
        which are required for creating the model.
    """

    def __init__(self, config: dict):
        super().__init__()

        self.config = config

        # ============
        # data props
        # ============
        data_props = config['data_props']
        self.num_historical_numeric = data_props.get('num_historical_numeric', 0)
        self.num_historical_categorical = data_props.get('num_historical_categorical', 0)
        self.historical_categorical_cardinalities = data_props.get('historical_categorical_cardinalities', [])

        self.num_static_numeric = data_props.get('num_static_numeric', 0)
        self.num_static_categorical = data_props.get('num_static_categorical', 0)
        self.static_categorical_cardinalities = data_props.get('static_categorical_cardinalities', [])

        self.num_future_numeric = data_props.get('num_future_numeric', 0)
        self.num_future_categorical = data_props.get('num_future_categorical', 0)
        self.future_categorical_cardinalities = data_props.get('future_categorical_cardinalities', [])

        self.historical_ts_representative_key = 'historical_ts_numeric' if self.num_historical_numeric > 0 \
            else 'historical_ts_categorical'
        self.future_ts_representative_key = 'future_ts_numeric' if self.num_future_numeric > 0 \
            else 'future_ts_categorical'

        # ============
        # model props
        # ============
        self.task_type = config.task_type
        self.attention_heads = config.model.attention_heads
        self.dropout = config.model.dropout
        self.lstm_layers = config.model.lstm_layers
        self.target_window_start_idx = (config.target_window_start - 1) if config.target_window_start is not None else 0
        if self.task_type == 'regression':
            self.output_quantiles = config.model.output_quantiles
            self.num_outputs = len(self.output_quantiles)
        elif self.task_type == 'classification':
            self.output_quantiles = None
            self.num_outputs = 1
        else:
            raise ValueError(f"unsupported task type: {self.task_type}")
        self.state_size = config.model.state_size

        # =====================
        # Input Transformation
        # =====================
        self.static_transform = InputChannelEmbedding(state_size=self.state_size,
                                                      num_numeric=self.num_static_numeric,
                                                      num_categorical=self.num_static_categorical,
                                                      categorical_cardinalities=self.static_categorical_cardinalities,
                                                      time_distribute=False)

        self.historical_ts_transform = InputChannelEmbedding(
            state_size=self.state_size,
            num_numeric=self.num_historical_numeric,
            num_categorical=self.num_historical_categorical,
            categorical_cardinalities=self.historical_categorical_cardinalities,
            time_distribute=True)

        self.future_ts_transform = InputChannelEmbedding(
            state_size=self.state_size,
            num_numeric=self.num_future_numeric,
            num_categorical=self.num_future_categorical,
            categorical_cardinalities=self.future_categorical_cardinalities,
            time_distribute=True)

        # =============================
        # Variable Selection Networks
        # =============================
        # %%%%%%%%%% Static %%%%%%%%%%%
        self.static_selection = VariableSelectionNetwork(
            input_dim=self.state_size,
            num_inputs=self.num_static_numeric + self.num_static_categorical,
            hidden_dim=self.state_size, dropout=self.dropout)

        self.historical_ts_selection = VariableSelectionNetwork(
            input_dim=self.state_size,
            num_inputs=self.num_historical_numeric + self.num_historical_categorical,
            hidden_dim=self.state_size,
            dropout=self.dropout,
            context_dim=self.state_size)

        self.future_ts_selection = VariableSelectionNetwork(
            input_dim=self.state_size,
            num_inputs=self.num_future_numeric + self.num_future_categorical,
            hidden_dim=self.state_size,
            dropout=self.dropout,
            context_dim=self.state_size)

        # =============================
        # static covariate encoders
        # =============================
        static_covariate_encoder = GatedResidualNetwork(input_dim=self.state_size,
                                                        hidden_dim=self.state_size,
                                                        output_dim=self.state_size,
                                                        dropout=self.dropout)
        self.static_encoder_selection = copy.deepcopy(static_covariate_encoder)
        self.static_encoder_enrichment = copy.deepcopy(static_covariate_encoder)
        self.static_encoder_sequential_cell_init = copy.deepcopy(static_covariate_encoder)
        self.static_encoder_sequential_state_init = copy.deepcopy(static_covariate_encoder)

        # ============================================================
        # Locality Enhancement with Sequence-to-Sequence processing
        # ============================================================
        self.past_lstm = nn.LSTM(input_size=self.state_size,
                                 hidden_size=self.state_size,
                                 num_layers=self.lstm_layers,
                                 dropout=self.dropout,
                                 batch_first=True)

        self.future_lstm = nn.LSTM(input_size=self.state_size,
                                   hidden_size=self.state_size,
                                   num_layers=self.lstm_layers,
                                   dropout=self.dropout,
                                   batch_first=True)

        self.post_lstm_gating = GateAddNorm(input_dim=self.state_size, dropout=self.dropout)

        # ============================================================
        # Static enrichment
        # ============================================================
        self.static_enrichment_grn = GatedResidualNetwork(input_dim=self.state_size,
                                                          hidden_dim=self.state_size,
                                                          output_dim=self.state_size,
                                                          context_dim=self.state_size,
                                                          dropout=self.dropout)

        # ============================================================
        # Temporal Self-Attention
        # ============================================================
        self.multihead_attn = InterpretableMultiHeadAttention(embed_dim=self.state_size, num_heads=self.attention_heads)
        self.post_attention_gating = GateAddNorm(input_dim=self.state_size, dropout=self.dropout)

        # ============================================================
        # Position-wise feed forward
        # ============================================================
        self.pos_wise_ff_grn = GatedResidualNetwork(input_dim=self.state_size,
                                                    hidden_dim=self.state_size,
                                                    output_dim=self.state_size,
                                                    dropout=self.dropout)
        self.pos_wise_ff_gating = GateAddNorm(input_dim=self.state_size, dropout=None)

        # ============================================================
        # Output layer
        # ============================================================
        self.output_layer = nn.Linear(self.state_size, self.num_outputs)

    def apply_temporal_selection(
        self,
        temporal_representation: torch.Tensor,
        static_selection_signal: torch.Tensor,
        temporal_selection_module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_samples, num_temporal_steps, _ = temporal_representation.shape

        # replicate the selection signal along time
        time_distributed_context = self.replicate_along_time(static_signal=static_selection_signal,
                                                             time_steps=num_temporal_steps)
        # Dimensions:
        # time_distributed_context: [num_samples x num_temporal_steps x state_size]
        # temporal_representation: [num_samples x num_temporal_steps x (total_num_temporal_inputs * state_size)]

        # for applying the same selection module on all time-steps, we stack the time dimension with the batch dimension
        temporal_flattened_embedding = self.stack_time_steps_along_batch(temporal_representation)
        time_distributed_context = self.stack_time_steps_along_batch(time_distributed_context)
        # Dimensions:
        # temporal_flattened_embedding: [(num_samples * num_temporal_steps) x (total_num_temporal_inputs * state_size)]
        # time_distributed_context: [(num_samples * num_temporal_steps) x state_size]

        # applying the selection module across time
        temporal_selection_output, temporal_selection_weights = temporal_selection_module.forward(
            temporal_flattened_embedding,
            time_distributed_context
        )
        # Dimensions:
        # temporal_selection_output: [(num_samples * num_temporal_steps) x state_size]
        # temporal_selection_weights: [(num_samples * num_temporal_steps) x (num_temporal_inputs) x 1]

        # Reshape the selection outputs and selection weights - to represent the temporal dimension separately
        temporal_selection_output = temporal_selection_output.view(num_samples, num_temporal_steps, -1)
        temporal_selection_weights = temporal_selection_weights.squeeze(-1).view(num_samples, num_temporal_steps, -1)
        # Dimensions:
        # temporal_selection_output: [num_samples x num_temporal_steps x state_size)]
        # temporal_selection_weights: [num_samples x num_temporal_steps x num_temporal_inputs)]

        return temporal_selection_output, temporal_selection_weights

    @staticmethod
    def replicate_along_time(static_signal: torch.Tensor, time_steps: int) -> torch.Tensor:
        """
        This method gets as an input a static_signal (non-temporal tensor) [num_samples x num_features],
        and replicates it along time for 'time_steps' times,
        creating a tensor of [num_samples x time_steps x num_features]

        Args:
            static_signal: the non-temporal tensor for which the replication is required.
            time_steps: the number of time steps according to which the replication is required.

        Returns:
            torch.tensor: the time-wise replicated tensor
        """
        # collapse any extra dims so shape == [batch, feat]
        ss = static_signal
        if ss.dim() > 2:
            ss = ss.flatten(start_dim=1)                # now [batch, feat]

        # unsqueeze into [batch, 1, feat] and repeat to [batch, time, feat]
        time_distributed_signal = ss.unsqueeze(1) \
                                   .repeat(1, time_steps, 1)
        return time_distributed_signal

    @staticmethod
    def stack_time_steps_along_batch(temporal_signal: torch.Tensor) -> torch.Tensor:
        """
        This method gets as an input a temporal signal [num_samples x time_steps x num_features]
        and stacks the batch dimension and the temporal dimension on the same axis (dim=0).

        The last dimension (features dimension) is kept as is, but the rest is stacked along dim=0.
        """
        return temporal_signal.view(-1, temporal_signal.size(-1))

    def transform_inputs(self,
                           static_feats_numeric: torch.Tensor,
                           static_feats_categorical: torch.Tensor,
                           historical_ts_numeric: torch.Tensor,
                           historical_ts_categorical: torch.Tensor,
                           future_ts_numeric: torch.Tensor,
                           future_ts_categorical: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This method processes input tensors and transforms each input channel (historical_ts, future_ts, static)
        separately to eventually return the learned embedding for each of the input channels

        each feature is embedded to a vector of state_size dimension:
        - numeric features will be projected using a linear layer
        - categorical features will be embedded using an embedding layer

        eventually the embedding for all the features will be concatenated together on the last dimension of the tensor
        (i.e. dim=1 for the static features, dim=2 for the temporal data).

        """
        static_rep = self.static_transform(x_numeric=static_feats_numeric,
                                           x_categorical=static_feats_categorical)
        historical_ts_rep = self.historical_ts_transform(x_numeric=historical_ts_numeric,
                                                         x_categorical=historical_ts_categorical)
        future_ts_rep = self.future_ts_transform(x_numeric=future_ts_numeric,
                                                 x_categorical=future_ts_categorical)
        return future_ts_rep, historical_ts_rep, static_rep

    def get_static_encoders(self, selected_static: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This method processes the variable selection results for the static data, yielding signals which are designed
        to allow better integration of the information from static metadata.
        Each of the resulting signals is generated using a separate GRN, and is eventually wired into various locations
        in the temporal fusion decoder, for allowing static variables to play an important role in processing.

        c_selection will be used for temporal variable selection
        c_seq_hidden & c_seq_cell will be used both for local processing of temporal features
        c_enrichment will be used for enriching temporal features with static information.
        """
        c_selection = self.static_encoder_selection(selected_static, context=None)
        c_enrichment = self.static_encoder_enrichment(selected_static, context=None)
        c_seq_hidden = self.static_encoder_sequential_state_init(selected_static, context=None)
        c_seq_cell = self.static_encoder_sequential_cell_init(selected_static, context=None)
        return c_enrichment, c_selection, c_seq_cell, c_seq_hidden

    def apply_sequential_processing(self, selected_historical: torch.Tensor, selected_future: torch.Tensor,
                                    c_seq_hidden: torch.Tensor, c_seq_cell: torch.Tensor) -> torch.Tensor:
        """
        This part of the model is designated to mimic a sequence-to-sequence layer which will be used for local
        processing.
        On that part the historical ("observed") information will be fed into a recurrent layer called "Encoder" and
        the future information ("known") will be fed into a recurrent layer called "Decoder".
        This will generate a set of uniform temporal features which will serve as inputs into the temporal fusion
        decoder itself.
        To allow static metadata to influence local processing, we use "c_seq_hidden" and "c_seq_cell" context vectors
        from the static covariate encoders to initialize the hidden state and the cell state respectively.
        The output of the recurrent layers is gated and fused with a residual connection to the input of this block.
        """

        # concatenate the historical (observed) temporal signal with the futuristic (known) temporal singal, along the
        # time dimension
        lstm_input = torch.cat([selected_historical, selected_future], dim=1)

        # the historical temporal signal is fed into the first recurrent module
        # using the static metadata as initial hidden and cell state
        # (initial cell and hidden states are replicated for feeding to each layer in the stack)

        # Prepare initial LSTM states: collapse any extra dims so shape=[batch, feat]
        h0 = c_seq_hidden
        c0 = c_seq_cell
        if h0.dim() > 2:
            h0 = h0.flatten(start_dim=1)   # [batch, feat]
        if c0.dim() > 2:
            c0 = c0.flatten(start_dim=1)

        # Unsqueeze to [1, batch, feat] and repeat across layers → [num_layers, batch, feat]
        h0 = h0.unsqueeze(0).repeat(self.lstm_layers, 1, 1)
        c0 = c0.unsqueeze(0).repeat(self.lstm_layers, 1, 1)

        past_lstm_output, hidden = self.past_lstm(
            selected_historical,
            (h0, c0)
        )
        # the future (known) temporal signal is fed into the second recurrent module
        # using the latest (hidden,cell) state of the first recurrent module
        # for setting the initial (hidden,cell) state.
        future_lstm_output, _ = self.future_lstm(selected_future, hidden)

        # concatenate the historical recurrent output with the futuristic recurrent output, along the time dimension
        lstm_output = torch.cat([past_lstm_output, future_lstm_output], dim=1)

        # perform gating to the recurrent output signal, using a residual connection to input of this block
        gated_lstm_output = self.post_lstm_gating(lstm_output, residual=lstm_input)
        return gated_lstm_output

    def apply_static_enrichment(self, gated_lstm_output: torch.Tensor,
                                static_enrichment_signal: torch.Tensor) -> torch.Tensor:
        """
        This static enrichment stage enhances temporal features with static metadata using a GRN.
        The static enrichment signal is an output of a static covariate encoder, and the GRN is shared across time.
        """
        num_samples, num_temporal_steps, _ = gated_lstm_output.shape

        # replicate the selection signal along time
        time_distributed_context = self.replicate_along_time(static_signal=static_enrichment_signal,
                                                             time_steps=num_temporal_steps)
        # Dimensions:
        # time_distributed_context: [num_samples x num_temporal_steps x state_size]

        # for applying the same GRN module on all time-steps, we stack the time dimension with the batch dimension
        flattened_gated_lstm_output = self.stack_time_steps_along_batch(gated_lstm_output)
        time_distributed_context = self.stack_time_steps_along_batch(time_distributed_context)
        # Dimensions:
        # flattened_gated_lstm_output: [(num_samples * num_temporal_steps) x state_size]
        # time_distributed_context: [(num_samples * num_temporal_steps) x state_size]

        # applying the GRN using the static enrichment signal as context data
        enriched_sequence = self.static_enrichment_grn(flattened_gated_lstm_output,
                                                       context=time_distributed_context)
        # Dimensions:
        # enriched_sequence: [(num_samples * num_temporal_steps) x state_size]

        # reshape back to represent temporal dimension separately
        enriched_sequence = enriched_sequence.view(num_samples, -1, self.state_size)
        # Dimensions:
        # enriched_sequence: [num_samples x num_temporal_steps x state_size]

        return enriched_sequence

    def apply_self_attention(self, enriched_sequence: torch.Tensor,
                             num_historical_steps: int,
                             num_future_steps: int):
        # create a mask - so that future steps will be exposed (able to attend) only to preceding steps
        output_sequence_length = num_future_steps - self.target_window_start_idx
        mask = torch.cat([torch.zeros(output_sequence_length,
                                      num_historical_steps + self.target_window_start_idx,
                                      device=enriched_sequence.device),
                          torch.triu(torch.ones(output_sequence_length, output_sequence_length,
                                                device=enriched_sequence.device),
                                     diagonal=1)], dim=1)
        # Dimensions:
        # mask: [output_sequence_length x (num_historical_steps + num_future_steps)]

        # apply the InterpretableMultiHeadAttention mechanism
        post_attention, attention_outputs, attention_scores = self.multihead_attn(
            q=enriched_sequence[:, (num_historical_steps + self.target_window_start_idx):, :],  # query
            k=enriched_sequence,  # keys
            v=enriched_sequence,  # values
            mask=mask.to(torch.bool))
        # Dimensions:
        # post_attention: [num_samples x num_future_steps x state_size]
        # attention_outputs: [num_samples x num_future_steps x state_size]
        # attention_scores: [num_samples x num_future_steps x num_total_steps]

        # Apply gating with a residual connection to the input of this stage.
        # Because the output of the attention layer is only for the future time-steps,
        # the residual connection is only to the future time-steps of the temporal input signal
        gated_post_attention = self.post_attention_gating(
            x=post_attention,
            residual=enriched_sequence[:, (num_historical_steps + self.target_window_start_idx):, :])
        # Dimensions:
        # gated_post_attention: [num_samples x num_future_steps x state_size]

        return gated_post_attention, attention_scores

    def forward(self,
                static_feats_numeric: torch.Tensor,
                static_feats_categorical: torch.Tensor,
                historical_ts_numeric: torch.Tensor,
                historical_ts_categorical: torch.Tensor,
                future_ts_numeric: torch.Tensor,
                future_ts_categorical: torch.Tensor) -> Dict[str, torch.Tensor]:
        # infer input structure - refactored for TorchScript compatibility
        if self.num_historical_numeric > 0:
            historical_rep_key = historical_ts_numeric
        else:
            historical_rep_key = historical_ts_categorical
        
        if self.num_future_numeric > 0:
            future_rep_key = future_ts_numeric
        else:
            future_rep_key = future_ts_categorical
        
        num_samples, num_historical_steps, _ = historical_rep_key.shape
        num_future_steps = future_rep_key.shape[1]
        # define output_sequence_length : num_future_steps - self.target_window_start_idx

        # =========== Transform all input channels ==============
        future_ts_rep, historical_ts_rep, static_rep = self.transform_inputs(
            static_feats_numeric=static_feats_numeric,
            static_feats_categorical=static_feats_categorical,
            historical_ts_numeric=historical_ts_numeric,
            historical_ts_categorical=historical_ts_categorical,
            future_ts_numeric=future_ts_numeric,
            future_ts_categorical=future_ts_categorical)
        # Dimensions:
        # static_rep: [num_samples x (total_num_static_inputs * state_size)]
        # historical_ts_rep: [num_samples x num_historical_steps x (total_num_historical_inputs * state_size)]
        # future_ts_rep: [num_samples x num_future_steps x (total_num_future_inputs * state_size)]

        # =========== Static Variables Selection ==============
        selected_static, static_weights = self.static_selection(static_rep, None)
        # Dimensions:
        # selected_static: [num_samples x state_size]
        # static_weights: [num_samples x num_static_inputs x 1]

        # =========== Static Covariate Encoding ==============
        c_enrichment, c_selection, c_seq_cell, c_seq_hidden = self.get_static_encoders(selected_static)
        # each of the static encoders signals is of shape: [num_samples x state_size]

        # =========== Historical variables selection ============
        # historical_ts_rep: (batch, time, state_size)
        batch_size, hist_steps, _ = historical_ts_rep.shape
        # flatten across time: (batch*time, state_size)
        hist_flat = historical_ts_rep.reshape(batch_size * hist_steps, -1)
        # repeat static selection for each time‐step
        # c_selection: (batch, state_size) → (batch, time, state_size) → (batch*time, state_size)
        # Ensure static selection is 2D: (batch, feat)
        cs = c_selection
        if cs.dim() > 2:
            # collapse any extra singleton/time dims into the feature axis
            cs = cs.flatten(start_dim=1)  # shape: [batch, feat]

        # repeat for each time‐step:
        #   (batch, feat) -> (batch, 1, feat) -> (batch, hist_steps, feat)
        cs = cs.unsqueeze(1).expand(batch_size, hist_steps, cs.size(1))
        # reshape to sequence of flattened samples: (batch*hist_steps, feat)
        # Flatten (batch, time, feat) → (batch*time, feat)
        static_ctx = cs.reshape(batch_size * hist_steps, cs.size(2))
        selected_historical, historical_selection_weights = self.historical_ts_selection.forward(
            hist_flat,
            static_ctx
        )
        # reshape back: (batch, time, state_size)
        selected_historical = selected_historical.view(batch_size, hist_steps, -1)


        # =========== Future variables selection ============
        # future_ts_rep: (batch, time, state_size)
        batch_size2, fut_steps, _ = future_ts_rep.shape
        # flatten across time
        fut_flat = future_ts_rep.reshape(batch_size2 * fut_steps, -1)
        # same static context for future steps
        # collapse any extra dims so c_selection is [batch, feat]
        cs2 = c_selection
        if cs2.dim() > 2:
            cs2 = cs2.flatten(start_dim=1)   # now (batch, feat)

        # unsqueeze & expand to (batch, fut_steps, feat)
        cs2 = cs2.unsqueeze(1).expand(batch_size2, fut_steps, cs2.size(1))

        # flatten to (batch*fut_steps, feat)
        static_ctx2 = cs2.reshape(batch_size2 * fut_steps, cs2.size(2))        # call the future selector
        selected_future, future_selection_weights = self.future_ts_selection.forward(
            fut_flat,
            static_ctx2
        )
        # reshape back: (batch, time, state_size)
        selected_future = selected_future.view(batch_size2, fut_steps, -1)

        # Dimensions:
        # selected_future: [num_samples x num_future_steps x state_size]
        # future_selection_weights: [num_samples x num_future_steps x total_num_future_inputs]

        # =========== Locality Enhancement - Sequential Processing ==============
        gated_lstm_output = self.apply_sequential_processing(selected_historical=selected_historical,
                                                             selected_future=selected_future,
                                                             c_seq_hidden=c_seq_hidden,
                                                             c_seq_cell=c_seq_cell)
        # Dimensions:
        # gated_lstm_output : [num_samples x (num_historical_steps + num_future_steps) x state_size]

        # =========== Static enrichment ==============
        enriched_sequence = self.apply_static_enrichment(gated_lstm_output=gated_lstm_output,
                                                         static_enrichment_signal=c_enrichment)
        # Dimensions:
        # enriched_sequence: [num_samples x (num_historical_steps + num_future_steps) x state_size]

        # =========== self-attention ==============
        gated_post_attention, attention_scores = self.apply_self_attention(enriched_sequence=enriched_sequence,
                                                                           num_historical_steps=num_historical_steps,
                                                                           num_future_steps=num_future_steps)
        # Dimensions:
        # attention_scores: [num_samples x output_sequence_length x (num_historical_steps + num_future_steps)]
        # gated_post_attention: [num_samples x output_sequence_length x state_size]

        # =========== position-wise feed-forward ==============
        # Applying an additional non-linear processing to the outputs of the self-attention layer using a GRN,
        # where its weights are shared across the entire layer
        post_poswise_ff_grn = self.pos_wise_ff_grn(gated_post_attention, None)
        # Also applying a gated residual connection skipping over the
        # attention block (using sequential processing output), providing a direct path to the sequence-to-sequence
        # layer, yielding a simpler model if additional complexity is not required
        gated_poswise_ff = self.pos_wise_ff_gating(
            post_poswise_ff_grn,
            residual=gated_lstm_output[:, (num_historical_steps + self.target_window_start_idx):, :])
        # Dimensions:
        # gated_poswise_ff: [num_samples x output_sequence_length x state_size]

        # =========== output projection ==============
        # Each predicted quantile has its own projection weights (all gathered in a single linear layer)
        predicted_quantiles = self.output_layer(gated_poswise_ff)
        # Dimensions:
        # predicted_quantiles: [num_samples x num_future_steps x num_quantiles]

        return {
            'predicted_quantiles': predicted_quantiles,
            'static_weights': static_weights.squeeze(-1),
            'historical_selection_weights': historical_selection_weights,
            'future_selection_weights': future_selection_weights,
            'attention_scores': attention_scores
        }
