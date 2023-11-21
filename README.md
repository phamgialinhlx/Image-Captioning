# <div align="center"> NLP: Image Captioning


## **1. Introduction**
Final project in the Natural Language Processing course.

Team Members: Trịnh Ngọc Huỳnh (20020054) and Phạm Gia Linh (20020203). 

## **2. Set Up**
  ### **Clone the repository**
    https://github.com/huynhspm/Image-Captioning
    
  ### **Install environment packages**
    cd Image-Captioning
    conda create -n image-caption python=3.10
    conda activate image-caption 
    pip install -r requirements.txt

  ### **Training**

  Set-up CUDA_VISIBLE_DEVICES and WANDB_API_KEY before training

  Configure the experiment using the available configuration in 'configs/experiment' or a custom configuration for your experiment 
  
    export CUDA_VISIBLE_DEVICES=???
    export WANDB_API_KEY=???
    python src/train.py experiment=<your_experiment> trainer.devices=<number_device>

 ### **Evaluation**
    export CUDA_VISIBLE_DEVICES=???
    python src/eval.py experiment=<your_experiment> ckpt_path=<your_checkpoint>

## **3. Results**


| Experiment Name | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
|-------|-------------|---------|-------------|------:|
| [rnn](configs/experiment/rnn.yaml) | 0.55241 | 0.37043 | 0.23651 | 0.1536 |
| [rnn_concat](configs/experiment/rnn_concat.yaml) | 0.54297 | 0.35271 | 0.22136 | 0.13712 |
| [lstm](configs/experiment/lstm.yaml) |  0.57097 | 0.38688 | 0.25114 | 0.16337 |
| [lstm_concat](configs/experiment/lstm_concat.yaml) | 0.56068 | 0.37466 | 0.23859 | 0.15176 |
| [transformer_encoder](configs/experiment/transformer_encoder.yaml) | 0.19013 | 0.10166 | 0.04107 | 0.0168 |
| [transformer](configs/experiment/transformer.yaml)  | 0.48475 | 0.29834 | 0.17457 | 0.10973 |
| [glove_transformer](configs/experiment/glove_transformer.yaml)  | 0.37894 | 0.20448 | 0.11482 | 0.07074 |

## 4. **Demo UI**
Specify your checkpoint in the 'ui.py' file for inference.

    python ui.py
![UI](images/ui.png)
