import torch
import numpy as np
from typing import List, Dict, Optional, Union, Literal

# DL modules
from models import (
    RNN_block,
    LSTM_block,
    GRU_block,
    self_attention_block,
    grouped_query_attention_block,
    xLSTM_block,
    combined_all_model_blocks
)
from deep_learning.loss_function import mse_loss, cross_entropy_loss

"""
本程式旨在建立深度學習模型於時間序列訓練與預測模組，包含：
1. 單純為時間序列
2. 時間序列加上單個時間點等資料
"""

class DL_time_series_training_flow():
    
    def __init__(
        self,
        num_of_time_series_sequences: int,
        num_of_time_series_features: int,
        DL_layers: List[Dict[str, Union[int, str]]],
        loss_func: Literal["mse", "cross_entropy"],
        optimizer: Literal["adam", "adamw"], 
        lr: float = 1e-3, 
        device: str = "cpu", 
        importance_methods: Optional[Literal["LIME"]] = None
    ):

        """
        Args: 
            - num_of_time_series_sequences (int)
            - num_of_time_series_features (int)
            - DL_layers (List[Dict[str, Union[int, str]]] | None)
        
        """

        self.device = device
        self.num_of_time_series_sequences = num_of_time_series_sequences
        self.num_of_time_series_features = num_of_time_series_features
        
        if DL_layers:
            self.DL_model = [
                self.call_model(i) for i in DL_layers
            ]
            self.DL_model = combined_all_model_blocks(model_list = self.DL_model)
            self.DL_model.to(device)

        if loss_func == "mse":
            from deep_learning.loss_function import mse_loss
            self.loss_func = mse_loss()

        elif loss_func == "cross_entropy":
            from deep_learning.loss_function import cross_entropy_loss
            self.loss_func = cross_entropy_loss

        if optimizer == "adam":
            from deep_learning.optimizer import adam_optimizer
            self.optimizer = adam_optimizer(
                model = self.DL_model,
                lr = lr
            )

        elif optimizer == "adamw":
            from deep_learning.optimizer import adamw_optimizer
            self.optimizer = adamw_optimizer(
                model = self.DL_model,
                lr = lr
            )
        return 
    
    def fit(
        self, 
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor]
    ):

        """
        X: 輸入資料，維度為 (batch_size, sequence_length, feature_size)，該資料是已經整理好且能夠被訓練的狀態，暫時不包含處理序列長度不一、有遺失值等議題
        """

        # Step1. 確認輸入資料是否已經包裝成 DataLoader，若不是的話請包裝
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)

        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        
        if y.shape.__len__() == 1: # 如果 y 是一維，則強制變成二維
            y = y.reshape(shape = (-1, 1))

        

        return 
    
    def call_single_model(self, one_model_dict: Dict[str, Union[int, str]]):

        if one_model_dict.keys()[0] in "RNN":
            return RNN_block(**one_model_dict)
        
        elif one_model_dict.keys()[0] in "LSTM":
            return LSTM_block(**one_model_dict)

        elif one_model_dict.keys()[0] in "GRU":
            return GRU_block(**one_model_dict)
        
    def call_multiple_model(self, one_multiply_model_dict: List[Dict[str, Union[int, str]]]):
        return 

    def model_training(self):
        return
    
    def model_evaluation(self):
        return
    
    def model_explanation(self):
        return 

class DL_time_series_and_cross_sectional_training_flow():
    
    def __init__(self):
        return
    
    def fit(self):
        return 