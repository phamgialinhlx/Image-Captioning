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
  ## **Demo UI**
  Specify your checkpoint in the 'ui.py' file for inference.

    python ui.py

![UI](images/ui.png)

