class Argument:
    
    # Setup
    cuda: bool = True
    save_path: str = '/project/graziul/ra/ywang27/saved_models/susas_cnn.pth'
    load_path: str = '/project/graziul/ra/ywang27/saved_models/no_text_no_merge_no_teo.pth'
    load_pretrained: bool = False
    train: bool = True
    evaluate: bool = True
    predict: bool = False
        
    # Feature
    delta_for_teo: int = 0.2
    
    # Dataset
    bpc_path: str = "/project/graziul/ra/team_ser/path/BPC_zone1_2018_08.csv"  # paths
    bpc_output_path: str = './labels/BPC_zone1_2018_08.csv'  # paths with labels
    bpc_feature_path: str = "./features/BPC_zone1_2018_08.json"

    susas_path: str = "./data/susas_path.csv"
    susas_feature_path: str = "./data/susas_feature.json"
    train_eval_split = 0.8  # For SUSAS only; train / (train + eval)
    
    # Label
    label_map: dict = {
        "HighPositive": 1,
        "LowPositive": 0,
        "HighNegative": 2
    }
    label_size = 3
        
    # Training
    batch_size: int = 16
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    epoch_num: int = 20
    num_workers: int = 0
    gradient_accumulate_step: int = 2
    
    # Model: FNN
    teo_feature_size: int = 1
    gemaps_feature_size: int = 25
    hidden_size = 1024
    adapt_layer_size: int = 256
    weight_mmd: float = 1

    # Model: CNN
    feature_size: int = 128
    window_sizes: list = [3, 4, 5]
    max_len_susas: int = 150
