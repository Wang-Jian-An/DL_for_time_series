import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
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
from deep_learning.training import DL_training
from deep_learning.metrics import (
    binary_classification_metrics,
    regression_metrics
)
from deep_learning.prediction import model_prediction

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
        epochs: int, 
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
        self.epochs = epochs
        self.basic_info = {
            "DL_layers": DL_layers,
            "loss_function": loss_func,
            "optimizer": optimizer,
            "epochs": epochs,
            "lr": lr
        }
        
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
        train_X: Union[np.ndarray, torch.Tensor],
        train_y: Union[np.ndarray, torch.Tensor],
        vali_X: Union[np.ndarray, torch.Tensor],
        vali_y: Union[np.ndarray, torch.Tensor], 
        test_X: Union[np.ndarray, torch.Tensor],
        test_y: Union[np.ndarray, torch.Tensor], 
        target_type: Literal["regression", "classification"], 
        batch_size: int = 16,
        num_workers: int = 4
    ):

        """
        X: 輸入資料，維度為 (batch_size, sequence_length, feature_size)，該資料是已經整理好且能夠被訓練的狀態，暫時不包含處理序列長度不一、有遺失值等議題
        """

        # Step1. 確認輸入資料是否已經包裝成 DataLoader，若不是的話請包裝
        train_X = torch.from_numpy(train_X) if isinstance(train_X, np.ndarray) else train_X
        vali_X = torch.from_numpy(vali_X) if isinstance(vali_X, np.ndarray) else vali_X
        test_X = torch.from_numpy(test_X) if isinstance(test_X, np.ndarray) else test_X

        train_y = torch.from_numpy(train_y) if isinstance(train_y, np.ndarray) else train_y
        vali_y = torch.from_numpy(vali_y) if isinstance(vali_y, np.ndarray) else vali_y
        test_y = torch.from_numpy(test_y) if isinstance(test_y, np.ndarray) else test_y
        
        train_y = train_y.reshape(shape = (-1, 1)) if train_y.size().__len__() == 1 else train_y
        vali_y = vali_y.reshape(shape = (-1, 1)) if vali_y.size().__len__() == 1 else vali_y
        test_y = test_y.reshape(shape = (-1, 1)) if test_y.size().__len__() == 1 else test_y

        # Step2. 把資料包裝成 DataLoader
        if isinstance(train_X, torch.Tensor) and isinstance(train_y, torch.Tensor):
            self.train_dataloader = DataLoader(
                TensorDataset(train_X, train_y), 
                batch_size = batch_size, 
                num_workers = num_workers
            )

        if isinstance(vali_X, torch.Tensor) and isinstance(vali_y, torch.Tensor):
            self.vali_dataloader = DataLoader(
                TensorDataset(vali_X, vali_y),
                batch_size = batch_size,
                num_workers = num_workers
            )

        if isinstance(test_X, torch.Tensor) and isinstance(test_y, torch.Tensor):
            self.test_dataloader = DataLoader(
                TensorDataset(test_X, test_y),
                batch_size = batch_size,
                num_workers = num_workers
            )

        # Step3. 進入迴圈、訓練模型
        self.DL_model, training_loss_list, vali_loss_list = DL_training(
            DL_model = self.DL_model,
            loss_func = self.loss_func,
            optimizer = self.optimizer,
            train_dataloader = self.train_dataloader,
            vali_dataloader = self.vali_dataloader,
            epochs = self.epochs
        )
        
        # 面對不同種類任務，給予不同流程
        evaluation_result = list()
        for one_set, one_data, one_target in zip(
            ["train", "vali", "test"],
            [self.train_dataloader, self.vali_dataloader, self.test_dataloader],
            [train_y, vali_y, test_y]
        ):

            # Step4. 模型預測
            yhat = model_prediction(
                DL_model = self.DL_model,
                input_data = one_data,
                target_type = target_type
            )            

            if target_type == "regression":

                # Step5. 模型評估
                one_eval_result = regression_metrics(
                    y_pred = yhat, y_true = one_target
                )

            elif target_type == "classification":
                
                # Step5. 模型評估（應該還要有判斷是否為二分類或多分類等）
                one_eval_result = binary_classification_metrics(
                    y_pred = yhat, y_true = one_target
                )

            # Step6. 儲存評估結果
            one_eval_result = {
                **self.basic_info,
                "Set": one_set,
                **one_eval_result
            }
            one_eval_result = one_eval_result.update(
                **{"Training Loss": training_loss_list} 
            ) if one_set == "train" else (
                one_eval_result.update(
                    **{"Validation": vali_loss_list}
                ) if one_set == "vali" else None
            )
            evaluation_result.append(one_eval_result)
        
        # Step7. 模型解釋


        return {
            "Evaluation": evaluation_result
        }
    
    def call_single_model(self, one_model_dict: Dict[str, Union[int, str]]):

        if one_model_dict.keys()[0] in "RNN":
            return RNN_block(**one_model_dict)
        
        elif one_model_dict.keys()[0] in "LSTM":
            return LSTM_block(**one_model_dict)

        elif one_model_dict.keys()[0] in "GRU":
            return GRU_block(**one_model_dict)
        
    def call_multiple_model(self, one_multiply_model_dict: List[Dict[str, Union[int, str]]]):
        return 
    
    def model_explanation(self):
        return 

class DL_time_series_and_cross_sectional_training_flow():
    
    def __init__(self):
        return
    
    def fit(self):
        return 