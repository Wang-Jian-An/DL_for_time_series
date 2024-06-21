import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, List

def combined_all_model_blocks(model_list):
    return nn.Sequential(model_list)

class model_interaction(nn.Module):
    def __init__(
        self,
        model_1,
        model_2,
        interaction_method: Literal["dot_product", "additive", "attention"] = "dot_product"
    ) -> None:
        super(model_interaction, self).__init__()

        self.model_1 = model_1
        self.model_2 = model_2
        
        if interaction_method == "dot_product":
            pass

        elif interaction_method == "additive":
            pass

        else:
            pass
        return
    
    def forward(self, X):


        return 

class RNN_block(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        nonlinearity: Literal["relu", "tanh"] = "tanh",
        bidirectional: bool = True
    ):
        super(RNN_block, self)
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
        
        h0 = torch.randn(size = (self.num_layers, X.size()[0], self.hidden_size))
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
        
        h0 = torch.randn(size = (self.num_layers, X.size()[0], self.hidden_size))
        c0 = torch.randn(size = (self.num_layers, X.size()[0], self.hidden_size))
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

        h0 = torch.randn(size = (self.num_layers, X.size()[0], self.hidden_size))
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

        self.each_compression_embed_size = target_length / target_compression_length

        self.embed_size = embed_size

        try:
            embed_size *= int(self.each_compression_embed_size)
        except: 
            assert "target_length 一定要整除於 target_compression_length"
                
        self.target_compression_length = target_compression_length
        self.model = nn.MultiheadAttention(
            embed_dim = embed_size,
            num_heads = num_heads
        )
        return
    
    def forward(self, X):

        X = X.reshape((-1, self.target_compression_length, self.each_compression_embed_size))
        X = self.model(X)
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
    
