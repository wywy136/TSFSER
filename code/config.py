class Argument:
    
    # Setup
    cuda: bool = False
        
    # Feature
    delta_for_teo: int = 0.2
    
    # Dataset
    bpc_path: str = "./data/BPC_path1.csv"
    bpc_num: int = 5000
    susas_path: str = "./data/susas_path.csv"
    
    # Label
    label_map: dict = {
        "HighPositive": 0,
        "LowPositive": 1,
        "HighNegative": 2
    }
    label_size = 3
        
    # Training
    batch_size: int = 2
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    epoch_num: int = 5
    num_workers: int = 0
    
    
    # Model
    teo_feature_size: int = 16
    gemaps_feature_size: int = 25
    adapt_layer_size: int = 20