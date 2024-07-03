import tqdm
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Dict, Optional, Union, Literal, Tuple

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

# DL prediction module
from deep_learning.prediction import model_prediction

# DL evaluation module


# CPU-GPU transition function
from deep_learning.utils import cpu_gpu_transition_for_pytorch

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
        DL_layers: List[Dict[str, Dict[str, Union[int, float, str]]]],
        loss_func: Union[Literal["mse", "cross_entropy"], List[Literal["mse", "cross_entropy"]]],
        optimizer: Literal["adam", "adamw"], 
        epochs: int, 
        lr: float = 1e-3, 
        device: str = "cpu", 
        importance_methods: Optional[Literal["LIME"]] = None
    ):

        """
        Args: 
            - num_of_time_series_sequences (int)
            - num_of_time_series_features (int)
            - DL_layers (List[Dict[str, Dict[str, Union[int, float, str]]]] | None)
        
        """

        self.epochs = epochs
        self.device = device
        self.num_of_time_series_sequences = num_of_time_series_sequences
        self.num_of_time_series_features = num_of_time_series_features
        self.target_type = "classification" if (
            loss_func in ["cross_entropy"]
        ) else (
            "regression"
        )
        
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
                model = self.DL_model.parameters(),
                lr = lr
            )

        elif optimizer == "adamw":
            from deep_learning.optimizer import adamw_optimizer
            self.optimizer = adamw_optimizer(
                model = self.DL_model.parameters(),
                lr = lr
            )
        return 
    
    def fit(
        self, 
        train_X: Union[np.ndarray, torch.Tensor],
        train_y: Union[np.ndarray, torch.Tensor],
        test_X: Union[np.ndarray, torch.Tensor],
        test_y: Union[np.ndarray, torch.Tensor],
        train_dataloader = None,
        test_dataloader = None,
        batch_size: Optional[int] = None
    ):

        """
        X: 輸入資料，維度為 (batch_size, sequence_length, feature_size)，該資料是已經整理好且能夠被訓練的狀態，暫時不包含處理序列長度不一、有遺失值等議題
        """

        # Step1. 確認輸入資料是否已經包裝成 DataLoader，若不是的話請包裝
        train_X = torch.from_numpy(train_X) if type(train_X) == np.ndarray else train_X
        train_y = torch.from_numpy(train_y) if type(train_y) == np.ndarray else train_y
        test_X = torch.from_numpy(test_X) if type(test_X) == np.ndarray else test_X
        test_y = torch.from_numpy(test_y) if type(test_y) == np.ndarray else test_y

        assert train_X.size()[0] == train_y.size()[0], "The number of train data and label must be the same. "
        assert test_X.size()[0] == test_y.size()[0], "The number of test data and label must be the same. "
        assert train_X.size()[-1] == train_y.size()[-1], "The number of features of train and test data must be the same. "

        train_y = train_y.reshape(shape = (-1, 1)) if train_y.shape.__len__() == 1 else train_y
        test_y = test_y.reshape(shape = (-1, 1)) if test_y.shape.__len__() == 1 else test_y

        train_dataloader = DataLoader(
            TensorDataset(train_X, train_y),
            batch_size = batch_size
        ) if not(train_dataloader) else train_dataloader
        test_dataloader = DataLoader(
            TensorDataset(test_X, test_y), 
            batch_size = batch_size
        ) if not(test_dataloader) else test_dataloader

        # Step2. 模型訓練
        model, train_loss_list, test_loss_list = self.model_training(
            model = self.DL_model,
            loss_func = self.loss_func,
            optimizer = self.optimizer, 
            train_dataloader = train_dataloader,
            test_dataloader = test_dataloader,
            epochs = self.epochs
        )

        # Step3. 模型評估
        

        # Step4. 確認是否儲存模型


        # Step5. Explainable AI (Optional)
        

        return 
    
    def call_model(
        self, 
        one_layer_info: Dict[str, Dict[str, Union[int, float, str]]]
    ):
        
        """
        Identify Single or multiple model. 
        """

        if one_layer_info.__len__() == 1:
            return self.call_single_model(
                one_model_dict = one_layer_info
            )

        else:
            return self.call_multiple_model(
                one_multiply_model_dict = one_layer_info
            )

    def call_single_model(
        self, 
        one_model_dict: Dict[str, Dict[str, Union[int, float, str]]]
    ):

        if one_model_dict.keys()[0] in "RNN":
            return RNN_block(**one_model_dict)
        
        elif one_model_dict.keys()[0] in "LSTM":
            return LSTM_block(**one_model_dict)

        elif one_model_dict.keys()[0] in "GRU":
            return GRU_block(**one_model_dict)
        
        elif one_model_dict.keys()[0] in "attention":
            return self_attention_block(**one_model_dict)
        
    def call_multiple_model(self, one_multiply_model_dict: List[Dict[str, Union[int, str]]]):
        return 

    def model_training(
        self,
        model: torch.nn.modules.Module,
        loss_func: torch.nn.modules.loss,
        optimizer: torch.optim,
        train_dataloader: torch.utils.data.dataloader.DataLoader,
        test_dataloader: torch.utils.data.dataloader.DataLoader,
        epochs
    ):
        
        """
        The process of deep learning model training, which is as follows. 
        1. 
        
        """

        train_loss_list = list()
        test_loss_list = list()

        for epoch in range(epochs):
            train_loss = list()
            test_loss = list()

            model.train()
            for X, target in train_dataloader:
                model, loss = self.model_training_block(
                    model = model,
                    loss_func = loss_func,
                    optimizer = optimizer,
                    X = X,
                    target = target,
                    device = self.device
                )
                train_loss.append(loss)

            model.eval()
            for X, target in test_dataloader:
                yhat = model_prediction(
                    model = model,
                    X = X,
                    device = self.device
                )
                loss = loss_func(yhat, target)
                test_loss.append(loss.item())

            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            print(
                "Epoch:", epoch, 
                "Train Loss:", sum(train_loss) / train_loss.__len__(),
                "Test Loss", sum(test_loss) / test_loss.__len__()
            )
        return model, train_loss_list, test_loss_list
    
    @cpu_gpu_transition_for_pytorch
    def model_training_block(
        self,
        model: torch.nn.modules.Module, 
        loss_func: torch.nn.modules.loss,
        optimizer: torch.optim, 
        X: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        target: torch.Tensor,
        device: str
    ) -> Tuple[torch.nn.modules.Module, Union[int, float]]:
        
        # # Put data into CPU or GPU
        # X.to(device) if (
        #     isinstance(X, torch.Tensor)
        # ) else (
        #     [i.to(device) for i in X]
        # )
        # target.to(device)
        
        # Model prediction
        yhat = model(X) if (
            isinstance(X, torch.Tensor)
        ) else (
            model(*X)
        )

        loss = loss_func(yhat, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # # Store loss
        # train_loss.append(loss.cpu().item)

        # # Remove data from GPU
        # X.cpu() if (
        #     isinstance(X, torch.Tensor)
        # ) else (
        #     [i.cpu() for i in X]
        # )
        # target.cpu()

        return model, loss.item()

    def model_evaluation(
        self,
        model: torch.nn.modules.Module,
        dataloader: torch.utils.data.dataloader.DataLoader
    ):
        
        # Model prediction
        yhat = list()
        target = list()
        for X, one_target in dataloader:
            yhat.append(
                model_prediction(
                    model = model,
                    dataloader = dataloader
                )
            )
            target.append(
                one_target
            )
        yhat = torch.Tensor(yhat)
        target = torch.Tensor(target)

        # Model evaluation

        return
    
    def model_explanation(self):
        return 

class DL_time_series_and_cross_sectional_training_flow():
    
    def __init__(self):
        return
    
    def fit(self):
        return 