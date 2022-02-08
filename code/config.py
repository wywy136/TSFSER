class Argument:
    
    # Setup
    cuda: bool = True
    save_path: str = '/project/graziul/ra/ywang27/saved_models/no_text_no_merge_no_teo_1.pth'
    load_path: str = '/project/graziul/ra/ywang27/saved_models/no_text_no_merge_no_teo.pth'
    bpc_output_path: str = './data/BPC_labels.csv'
    load_pretrained: bool = True
    train: bool = True
    evaluate: bool = False
    predict: bool = False
        
    # Feature
    delta_for_teo: int = 0.2
    
    # Dataset
    bpc_path: str = "./data/BPC_path1.csv"
    bpc_num: int = 5000
    susas_path: str = "./data/susas_path.csv"
    train_test_split = 1
    
    # Label
    label_map: dict = {
        "HighPositive": 1,
        "LowPositive": 0,
        "HighNegative": 2
    }
    label_size = 3
        
    # Training
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    epoch_num: int = 5
    num_workers: int = 0
    gradient_accumulate_step: int = 1
    
    # Model
    teo_feature_size: int = 1
    gemaps_feature_size: int = 25
    adapt_layer_size: int = 20
    weight_mmd: float = 1