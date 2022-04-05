This repository contains the code for the manuscript "Transfer Learning on Stress Related Speech Emotion Recognition"

- `/data`: dataset for this project, includes BPC and SUSAS
- `/dataset`: Dataset class, Collator class for BPC and SUSAS
- `/evaluator`: Evaluator class for evaluation of SUSAS
- `/feature_extractor`: Feature Extractor class for GeMAPS and TEO
- `/features`: extracted features for BPC
- `/labels`: predicted labels for BPC
- `/loss`: Classification Loss class and MMD Loss class
- `/model`: FNN, CNN class for the model
- `/predictor`: Predictor class for prediction on BPC
- `/trainer`: Trainer class for the training procedure wrapping all the other classes
- `/config.py`: Argument class for configuration of the program