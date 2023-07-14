# Project5: Multimodal Sentiment Classification

AI(23Spring)  Project5: Multimodal Sentiment Classification

## Set Up

```bash
pip install -r requirements.txt
```

## Code Structure

```sh
|-- data/
    |-- input/                          # Preprocessed data
        |-- dev_data.json               # Preprocessed evaluate data
        |-- test_data.json              # Preprocessed test data
        |-- train_data.json             # Preprocessed train data
    |-- test_without_label.txt          # Test data with GUID and empty labels
    |-- train.txt                       # Training data with GUID and labels
|-- model/
    |-- pretrain_model.py               # ViT & BERT
    |-- img_classification.py           # Model (image only)
    |-- text_classification.py          # Model (text only)
    |-- multi_classification.py         # Model (multimodal)
|-- result/
    |-- test_with_label.txt             # prediction file
|-- utils/
	|-- config.py                       # General configuration file
	|-- config_util.py                  # Update configuration with arguments
	|-- data_util.py                    # Data preprocessing related files
	|-- img_util.py                     # Image Dataset generator
    |-- text_util.py                    # Text Dataset generator	
	|-- multimodel_data_util.py         # Image & text Dataset generator
	|-- parser.py                       # Parse arguments
	|-- plt.py                          # Plot statistical charts 
    |-- run_util.py                      # Training and other utility methods
|-- README.md              
|-- requirements.txt               # Model (text only)
|-- run.py                              # Execution entry file
```

## Model Usage

1. Put the experimental data in the `data/raw/` directory (optional, if the dataset is not in this directory, you need to manually specify `--raw_data_path` when running `run.py`).

2. Run the `run.py` file.

   Train and test (using default parameters):

   ```bash
   python run.py --train --test
   ```

   Predict (using default parameters):

   ```bash
   python run.py --predict
   ```

   

