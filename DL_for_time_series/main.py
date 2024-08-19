import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Dict, Optional, Union, Literal, List

import torch.utils.data
import torch.utils.data.dataloader

# DL modules
from .models import define_model as define_model_for_time_series
from .models import time_series_concatenate_cross_sectional_model
from deep_learning import define_model as define_model_for_general_data
from deep_learning import call_model as call_model_for_general_data
from deep_learning import model_training
from deep_learning import model_prediction
from deep_learning import (
    binary_classification_metrics,
    multiclass_classification_metrics,
    regression_metrics
)
from monitor.folder import folder_exists

"""
本程式旨在建立深度學習模型於時間序列訓練與預測模組，包含：
1. 單純為時間序列
2. 時間序列加上單個時間點等資料
"""

class DL_time_series_training_flow:
    
    def __init__(
        self,
        # num_of_time_series_sequences: int,
        # num_of_time_series_features: int,
        DL_layers: List[Dict[str, Dict[str, Union[int, float, str]]]],
        loss_func: Literal["mse", "cross_entropy"],
        optimizer: Literal["adam", "adamw"], 
        epochs: int, 
        target_type: Literal["binary_classification", "multiclass_classification", "regression"], 
        model_name: str = "sample_model", 
        lr: float = 1e-3, 
        device: str = "cpu", 
        folder_path: Optional[str] = None, 
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
        # self.num_of_time_series_sequences = num_of_time_series_sequences
        # self.num_of_time_series_features = num_of_time_series_features
        self.model_name = model_name
        self.target_type = target_type
        self.folder_path = folder_path
        
        if DL_layers:
            self.model = define_model_for_time_series(
                model_name = model_name,
                model_list = DL_layers
            )
            self.model.to(device)

        print(self.model)

        if loss_func == "mse":
            from deep_learning import mse_loss
            self.loss_func = mse_loss()

        elif loss_func == "cross_entropy":
            from deep_learning import cross_entropy_loss
            self.loss_func = cross_entropy_loss

        if optimizer == "adam":
            from deep_learning import adam_optimizer
            self.optimizer = adam_optimizer(
                model = self.model,
                lr = lr
            )

        elif optimizer == "adamw":
            from deep_learning import adamw_optimizer
            self.optimizer = adamw_optimizer(
                model = self.model,
                lr = lr
            )
        return 
    
    def fit(
        self, 
        train_X: Optional[
            Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]]
        ] = None,
        train_y: Optional[Union[np.ndarray, torch.Tensor]] = None,
        test_X: Optional[
            Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]]
        ] = None,
        test_y: Optional[Union[np.ndarray, torch.Tensor]] = None,
        train_dataloader: torch.utils.data.dataloader.DataLoader = None,
        test_dataloader: torch.utils.data.dataloader.DataLoader = None,
        batch_size: Optional[int] = None
    ):

        """
        X: 輸入資料，維度為 (batch_size, sequence_length, feature_size)，該資料是已經整理好且能夠被訓練的狀態，暫時不包含處理序列長度不一、有遺失值等議題
        """

        assert (
            not(train_X is None) and not(train_y is None)
        ) or (
            not(train_dataloader is None)
        ), "Either tensor or dataloader must be existed for train set."
        assert (
            not(test_X is None) and not(test_y is None)
        ) or (
            not(test_dataloader is None)
        ), "Either tensor or dataloader must be existed for test set."

        # Step1. 確認資料格式，應為 DataLoader
        if not(train_X is None) and not(train_y is None) and not(train_dataloader):
            if not(type(train_X) == list):
                train_X = [train_X]

            train_y = (
                torch.from_numpy(train_y).float()
            ) if type(train_y) == np.ndarray else train_y.float()
            for index in range(train_X.__len__()):
                train_X[index] = (
                    torch.from_numpy(train_X[index]).float()
                ) if type(train_X[index]) == np.ndarray else train_X[index].float()
                assert train_X[index].size()[0] == train_y.size()[0], "The number of train data and label must be the same. "
            
            train_y = train_y.reshape(shape = (-1, 1)) if train_y.shape.__len__() == 1 else train_y

        if not(test_X is None) and not(test_y is None) and not(test_dataloader):
            if not(type(test_X) == list):
                test_X = [test_X]
            
            test_y = torch.from_numpy(test_y).float() if type(test_y) == np.ndarray else test_y.float()
            for index in range(test_X.__len__()):
                test_X[index] = (
                    torch.from_numpy(test_X[index]).float()
                ) if type(test_X[index]) == np.ndarray else test_X[index].float()
                assert test_X[index].size()[0] == test_y.size()[0], "The number of test data and label must be the same. "
            
            test_y = test_y.reshape(shape = (-1, 1)) if test_y.shape.__len__() == 1 else test_y
  
        # assert train_X.size()[-1] == test_X.size()[-1], "The number of features of train and test data must be the same. "

        train_dataloader = DataLoader(
            TensorDataset(*train_X, train_y),
            batch_size = batch_size
        ) if not(train_dataloader) else train_dataloader
        test_dataloader = DataLoader(
            TensorDataset(*test_X, test_y), 
            batch_size = batch_size
        ) if not(test_dataloader) else test_dataloader

        # Step2. 模型訓練
        self.model, train_loss_list, test_loss_list = model_training(
            model = self.model,
            loss_func = self.loss_func,
            optimizer = self.optimizer,
            train_dataloader = train_dataloader,
            test_dataloader = test_dataloader
        )

        # Step3. 模型評估
        evaluation_result = list()
        for set_name, set, loss_list in zip(
            ["train", "test"],
            [train_dataloader, test_dataloader],
            [train_loss_list, test_loss_list]
        ):
            evaluation_result.append(
                self.model_evaluation(
                    set = set_name,
                    dataloader = set,
                    loss_for_each_epoch = loss_list
                )
            )
        

        # Step4. 儲存模型
        self.model_storage(
            folder_path = self.folder_path
        ) if self.folder_path else None


        # Step5. Explainable AI (Optional)
        

        return evaluation_result

    @folder_exists
    def model_storage(
        self,
        folder_path
    ):
        
        torch.save(
            self.model.state_dict(), 
            os.path.join(
                folder_path,
                f"{self.model_name}.pth"   
            )
        )
        return 

    def model_evaluation(
        self,
        set: Literal["train", "validation", "test"], 
        dataloader: torch.utils.data.dataloader,
        **kwargs
    ) -> Dict[str, Union[str, int, float, List[float]]]:

        """
        在此要輸出模型評估表中「一筆」評估結果，內容包含：
        1. 模型名稱
        2. 特徵工程方法
        3. 訓練、驗證還是測試資料集
        4. 資料筆數
        5. 評估結果
        """

        # Model prediction
        target_list = list()
        yhat_proba_list = list()
        for data in dataloader:
            X, target = data[:-1], data[-1] # Only the last one is assigned to be a target.
            X = X[0] if X.__len__() == 1 else X
            yhat_proba = model_prediction(
                model = self.model,
                X = X
            )
            target_list.append(target)
            yhat_proba_list.append(yhat_proba)

        # Clean prediction result and its true value
        target = torch.vstack(target_list)
        target = target.flatten()
        yhat_proba = torch.vstack(yhat_proba_list)
        yhat = torch.argmax(yhat_proba, dim = -1)

        # Compute some of metrics for either classification or regression
        evaluation_result = binary_classification_metrics(
            yhat = yhat,
            yhat_proba = yhat_proba,
            target = target
        ) if self.target_type == "binary_classification" else (
            multiclass_classification_metrics(
                yhat = yhat,
                target = target
            ) if self.target_type == "multiclass_classification" else (
                regression_metrics(
                    yhat = yhat,
                    target = target
                )
            )
        )

        # Find the number of data for each class
        num_of_data = {
            "num_of_data": target.__len__()
        } if self.target_type == "regression" else {
            "num_of_{}".format(int(key)): value
            for key, value in pd.Series(target).value_counts().to_dict()
        }
        return {
            "model": self.model_name,
            "set": set,
            **num_of_data,
            **evaluation_result,
            **kwargs
        }
    
    def model_explanation(self):
        return 

class DL_time_series_and_cross_sectional_training_flow:
    
    def __init__(
        self,
        DL_time_series_layers: List[Dict[str, Union[int, float, List[Dict[str, Union[int, float]]]]]],
        DL_NN_layers: List[Dict[str, Union[int, float, List[Dict[str, Union[int, float]]]]]],
        DL_decoder_layers: List[Dict[str, Union[int, float, List[Dict[str, Union[int, float]]]]]],
        loss_func: Literal["mse", "cross_entropy"],
        optimizer: Literal["adam", "adamw"],
        lr: float,
        model_name: str, 
        folder_path: str = None
    ):
        
        self.folder_path = folder_path
        self.model_name = model_name

        # Define time series model
        time_series_model = define_model_for_time_series(
            model_name = "time_series_model",
            model_list = DL_time_series_layers
        )

        # Define tabular model
        nn_model = define_model_for_general_data(
            model_name = "nn_model",
            model_list = [
                call_model_for_general_data(
                    one_layer_dict = one_layer_dict
                )
                for one_layer_dict in DL_NN_layers
            ]
        )

        # Define decoder model
        decoder_model = define_model_for_general_data(
            model_name = "decoder_model",
            model_list = [
                call_model_for_general_data(
                    one_layer_dict = one_layer_dict
                )
                for one_layer_dict in DL_decoder_layers
            ]
        )

        # Define model
        self.model = time_series_concatenate_cross_sectional_model(
            DL_time_series_model = time_series_model,
            DL_NN_model = nn_model,
            DL_decoder_model = decoder_model
        )

        # Define loss function and optimizer
        if loss_func == "mse":
            from deep_learning import mse_loss
            self.loss_func = mse_loss()

        elif loss_func == "cross_entropy":
            from deep_learning import cross_entropy_loss
            self.loss_func = cross_entropy_loss

        if optimizer == "adam":
            from deep_learning import adam_optimizer
            self.optimizer = adam_optimizer(
                model = self.model,
                lr = lr
            )

        elif optimizer == "adamw":
            from deep_learning import adamw_optimizer
            self.optimizer = adamw_optimizer(
                model = self.model,
                lr = lr
            )

        return
    
    def fit(
        self, 
        train_dataloader: torch.utils.data.dataloader.DataLoader,
        test_dataloader: torch.utils.data.dataloader.DataLoader,
        batch_size: Optional[int] = None
    ):

        """
        X: 輸入資料，維度為 (batch_size, sequence_length, feature_size)，該資料是已經整理好且能夠被訓練的狀態，暫時不包含處理序列長度不一、有遺失值等議題
        """

        # Step2. 模型訓練
        self.model, train_loss_list, test_loss_list = model_training(
            model = self.model,
            loss_func = self.loss_func,
            optimizer = self.optimizer,
            train_dataloader = train_dataloader,
            test_dataloader = test_dataloader
        )

        # Step3. 模型評估
        evaluation_result = list()
        for set_name, set, loss_list in zip(
            ["train", "test"],
            [train_dataloader, test_dataloader],
            [train_loss_list, test_loss_list]
        ):
            evaluation_result.append(
                self.model_evaluation(
                    set = set_name,
                    dataloader = set,
                    loss_for_each_epoch = loss_list
                )
            )
        

        # Step4. 儲存模型
        self.model_storage(
            folder_path = self.folder_path
        ) if self.folder_path else None


        # Step5. Explainable AI (Optional)
        

        return evaluation_result

    @folder_exists
    def model_storage(
        self,
        folder_path
    ):
        
        torch.save(
            self.model.state_dict(), 
            os.path.join(
                folder_path,
                f"{self.model_name}.pth"   
            )
        )
        return 

    def model_evaluation(
        self,
        set: Literal["train", "validation", "test"], 
        dataloader: torch.utils.data.dataloader,
        **kwargs
    ) -> Dict[str, Union[str, int, float, List[float]]]:

        """
        在此要輸出模型評估表中「一筆」評估結果，內容包含：
        1. 模型名稱
        2. 特徵工程方法
        3. 訓練、驗證還是測試資料集
        4. 資料筆數
        5. 評估結果
        """

        # Model prediction
        target_list = list()
        yhat_proba_list = list()
        for data in dataloader:
            X, target = data[:-1], data[-1] # Only the last one is assigned to be a target.
            X = X[0] if X.__len__() == 1 else X
            yhat_proba = model_prediction(
                model = self.model,
                X = X
            )
            target_list.append(target)
            yhat_proba_list.append(yhat_proba)

        # Clean prediction result and its true value
        target = torch.vstack(target_list)
        target = target.flatten()
        yhat_proba = torch.vstack(yhat_proba_list)
        yhat = torch.argmax(yhat_proba, dim = -1)

        # Compute some of metrics for either classification or regression
        evaluation_result = binary_classification_metrics(
            yhat = yhat,
            yhat_proba = yhat_proba,
            target = target
        ) if self.target_type == "binary_classification" else (
            multiclass_classification_metrics(
                yhat = yhat,
                target = target
            ) if self.target_type == "multiclass_classification" else (
                regression_metrics(
                    yhat = yhat,
                    target = target
                )
            )
        )

        # Find the number of data for each class
        num_of_data = {
            "num_of_data": target.__len__()
        } if self.target_type == "regression" else {
            "num_of_{}".format(int(key)): value
            for key, value in pd.Series(target).value_counts().to_dict()
        }
        return {
            "model": self.model_name,
            "set": set,
            **num_of_data,
            **evaluation_result,
            **kwargs
        }
    
    def model_explanation(self):
        return 