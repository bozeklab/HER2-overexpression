# HER2-overexpression

Prediction of HER2 overexpression status and IHC score as seen in "Predicting the HER2 status in esophageal cancer from tissue microarrays using convolutional neural networks".

## Usage:  

### Quick intro
To train a resnet on example images available in _data/example_images_ run:  
```console
$ conda env create --file environment.yml
$ conda activate her2
$ python3 train.py
```

Similarly, to train the MIL classifier:
```console
$ python3 train.py --model=abmil --batch_size=1 --img_size=5468
```

See the information below on how to correctly use the available scripts.

### Training with Google Colab
 **train.py** makes use of **CUDA** to accelerate the training of the model. In case CUDA is not available to you, we provide **train_notebook.ipynb** that can be open with [**Google Colab**](https://colab.research.google.com/)

### Data format
Each dataset should be in a .csv file with the following columns:
- **filename**: paths of the images. 
- **ihc-score**: Numeric labels of the IHC score (0..3)
- **her2-status**: Numeric labels of the overexpression status (0 or 1)
- optionally, a column **weigthed_sampler_label** with numeric labels that could be used for the weighted sampling when training the CNNs (in our work, 0..4 having score and status 2- and 2+ as two separate classes)

### Scripts:
**train.py**: Train the CNNs. Arguments:  
  --model MODEL         resnet34 or abmil (default: resnet34).  
  --task TASK           ihc-score or her2-status (default: her2-status).  
  --weighted_sampler_label WEIGHTED_SAMPLER_LABEL
                        Additional label in the train .csv to weight the sampling (default: None).  
  --gpus GPUS           Number of GPUs (default: 4).  
  --epochs EPOCHS       Number of epochs (default: 100).  
  --learning_rate [LEARNING_RATE]
                        Learning rate (default: 0.001).  
  --scheduler_factor [SCHEDULER_FACTOR]
                        Scheduler factor for decreasing learning rate (default: 0.1).  
  --scheduler_patience [SCHEDULER_PATIENCE]
                        Scheduler patience for decreasing learning rate (default: 10).  
  --batch_size [BATCH_SIZE]
                        Batch size (default: 32).  
  --train_csv TRAIN_CSV
                        .csv file containing the training examples (default: train.csv).  
  --val_csv VAL_CSV     .csv file containing the val examples (default: None).  
  --checkpoints_dir CHECKPOINTS_DIR
                        Path to save model checkpoints (default: ./checkpoints).  
  --ip_address MASTER_ADDR
                        IP address of rank 0 node (default: localhost).  
  --port MASTER_PORT    Free port on rank 0 node (default: 8888).  
  --num_workers NUM_WORKERS
                        Number of workers for loading data (default: 0).  
  --img_size IMG_SIZE   Input image size for the model (default: 1024).  
  
  **inference.py**: Run a trained CNN on a test dataset, and export a .csv with the class probabilities for each image. Arguments:  
  --model MODEL         resnet34 or abmil (default: resnet34).  
  --task TASK           ihc-score or her2-status (default: her2-status).  
  --batch_size [BATCH_SIZE]
                        Batch size (default: 32).  
  --test_csv TEST_CSV   .csv file containing the test examples (default: test.csv).  
  --checkpoint_dir CHECKPOINT_DIR
                        Path to load the model checkpoint from (default: ./checkpoints/checkpoint_99.pth.tar).  
  --out_dir OUT_DIR     Path to save the inference results (default: ./out).  
  --num_workers NUM_WORKERS
                        Number of workers for loading data (default: 0).  
  --img_size IMG_SIZE   Input image size for the model (default: 1024).  
  
  **staining_intensity_classifier.py**: Train and test the staining intensity classifier. Arguments:  
  --task TASK           ihc-score or her2-status (default: her2-status).  
  --train_csv TRAIN_CSV
                        .csv file containing the training examples (default: train.csv).  
  --test_csv TEST_CSV   .csv file containing the test examples (default: None).  
  --pickle_dir PICKLE_DIR
                        Folder to store the pickled classifier (default: ./).  
  --out_dir OUT_DIR     Path to save the inference results (default: ./out).  
