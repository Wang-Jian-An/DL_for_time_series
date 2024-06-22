# DL_for_time_series

## 模型輸入格式
- Model
```json
[
    {
        "RNN": ...
    },
    {
        "LSTM": ...
    },
    {
        "GRU": ...,
        "LSTM": ...
    }
]
```

- RNN
```json
{
    "RNN": {
        "input_size": int, 
        "hidden_size": int,
        "num_layers": int,
        "nonlinearity": str,
        "bidirectional": bool
    }
}
```

- LSTM
```json
{
    "LSTM": {
        "input_size": int,
        "hidden_size": int,
        "num_layers": int,
        "bidirectional": bool
    }
}
```

- GRU
```json
{
    "GRU": {
        "input_size": int,
        "hidden_size": int,
        "num_layers": int,
        "bidirectional": bool
    }
}
```

- Attention
```json
{
    "Attention": {
        "embed_size": int,
        "target_length": int,
        "target_compression_length": float,
        "num_heads" int
    }
}
```