import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, List, Dict, Union
from deep_learning import define_model as define_model_for_general_data
from deep_learning import call_model as call_model_for_general_data

with open(os.path.join(os.getcwd(), "config.yaml")) as f:
    config = yaml.safe_load(f)

def define_model(
    model_name: str, 
    model_list: List[
        Dict[str, Union[List[Dict[str, Dict[str, Union[str, int, float]]]], Dict[str, Union[str, int, float]]]]
    ]
) -> torch.nn.modules.Module:
    
    """
    Define deep learning model. 
    """

    class CustomModel(nn.Module):
        def __init__(
            self,
            model_list: List[torch.nn.modules.Module]
        ):
            super(CustomModel, self).__init__()
            self.model = nn.Sequential(*model_list)
            return 
        
        def forward(self, X):
            return self.model(X)

    model_list = [
        call_model(
            one_layer_dict = one_layer_dict
        )
        for one_layer_dict in model_list
    ]
    CustomModel.__name__ = model_name
    model = CustomModel(
        model_list = model_list
    )
    return model

def call_model(
    one_layer_dict: Dict[str, Dict[str, Union[str, int]]]
) -> torch.nn.modules.Module:
    
    """
    Define one of the layers in the DL model. 
    """

    layer_name, layer_parameter = next(iter(one_layer_dict.items()))

    assert layer_name in config["layer_name_list"], "Layer must be one of the following: {}".format(config["layer_name_list"])

    if layer_name == "RNN":
        return RNN_block(**layer_parameter)

    elif layer_name == "LSTM":
        return LSTM_block(**layer_parameter)

    elif layer_name == "GRU":
        return GRU_block(**layer_parameter)

    elif layer_name == "self-attention":
        return self_attention_block(**layer_parameter)

    elif layer_name in ["linear", "flatten"]:
        return call_model_for_general_data(
            one_layer_dict = one_layer_dict
        ).model
    
    # elif layer_name == "flatten":
    #     return 

    elif layer_name == "KAN":
        pass

    elif layer_name == "residual":
        pass

class time_series_concatenate_cross_sectional_model(nn.Module):
    def __init__(
        self,
        DL_time_series_model: torch.nn.modules.Module,
        DL_NN_model: torch.nn.modules.Module,
        DL_decoder_model: torch.nn.modules.Module
    ):
        super(time_series_concatenate_cross_sectional_model, self).__init__()

        self.model = nn.ModuleDict(
            time_series_model = DL_time_series_model,
            nn_model = DL_NN_model,
            decoder_model = DL_decoder_model
        )

        return 
    
    def forward(
        self,
        time_series_data: torch.Tensor,
        cross_sectional_data: torch.Tensor
    ):
        
        time_series_data = self.model["time_series_model"](time_series_data)
        cross_sectional_data = self.model["nn_model"](cross_sectional_data)

        x = torch.concat([time_series_data, cross_sectional_data], axis = 0)

        x = self.model["decoder_model"](x)

        return x

class RNN_block(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        nonlinearity: Literal["relu", "tanh"] = "tanh",
        bidirectional: bool = True
    ):
        super(RNN_block, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.model = nn.RNN(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            nonlinearity = nonlinearity,
            batch_first = True,
            bidirectional = bidirectional
        )
        return 
    
    def forward(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        
        h0 = torch.randn(size = (self.num_layers * 2, X.size()[0], self.hidden_size))
        output, h = self.model(X, h0)
        return output
    
class LSTM_block(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool = True
    ):
        super(LSTM_block, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.model = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            bidirectional = bidirectional,
            batch_first = True
        )
        return 
    
    def forward(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        
        h0 = torch.randn(size = (self.num_layers * 2, X.size()[0], self.hidden_size))
        c0 = torch.randn(size = (self.num_layers * 2, X.size()[0], self.hidden_size))
        output, (h, c) = self.model(X, (h0, c0))
        return output
    
class GRU_block(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool = True
    ):
        super(GRU_block, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.model = nn.GRU(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            bidirectional = bidirectional,
            batch_first = True
        )
        return
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:

        h0 = torch.randn(size = (self.num_layers * 2, X.size()[0], self.hidden_size))
        output, h = self.model(X, h0)        
        return
    
class self_attention_block(nn.Module):
    def __init__(
        self,
        embed_size: int,
        target_length: int,
        target_compression_length: float,
        num_heads: int
    ):
        super(self_attention_block, self).__init__()

        """
        
        Arguments 
        -------------------
        embed_size (int)
            - The number of features about each time point. 

        target_length (int)
            - The number of data in the sequence. 
        
        target_compression_length (float)
            - 

        """

        assert float(int(target_length / target_compression_length)) == target_length / target_compression_length, "target_length 一定要整除於 target_compression_length"

        self.each_compression_embed_size = int(target_length / target_compression_length)

        self.embed_size = embed_size
        embed_size *= self.each_compression_embed_size
                
        self.target_compression_length = target_compression_length
        self.model = nn.MultiheadAttention(
            embed_dim = embed_size,
            num_heads = num_heads,
            batch_first = True
        )
        return
    
    def forward(
        self, 
        X: torch.Tensor
    ):

        X = X.reshape((-1, self.target_compression_length, self.each_compression_embed_size * self.embed_size))
        X, _ = self.model(
            query = X,
            key = X,
            value = X
        )
        return X.reshape((X.size()[0], -1, self.embed_size))

class grouped_query_attention_block(nn.Module):
    def __init__(self):
        super(grouped_query_attention_block, self).__init__()
        return
    
    def forward(self, X):
        return 

class transformer_block(nn.Module):
    def __init__(self):
        super(transformer_block, self).__init__()
        return
    
    def forward(self, X):
        return
    
class xLSTM_block(nn.Module):
    def __init__(self):
        super(xLSTM_block, self).__init__()
        return
    
    def forward(self, X):
        return
    
