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
        
    # Training
    batch_size: int = 2
    epoch_num: int = 5
    num_workers: int = 0
    