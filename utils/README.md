### Readme for the utils

The `data_processing.py` is a misc with some functions to convert the trajectories to and from ARDL typical .json format

The other two folders contain the functions for pre-training eigenvalue converter and success rate predictor, in case if you would like some examples of how to do it from scratch. There are some training data provided for both pre-trainings, they are more of an example, than the exact same as in the paper (due to space constraints), please don't forget to unpack the archives before running the codes.

**Please note that pre-trained versions of these models are already included in the master folder named `model_0_50k_epochs.pth` and `model_0_SR.pth` correspondingly.**


### Prerequisites
You might additionally need `re, matplotlib, and json` packages to run the contents of this directory.
That is in addition to the prerequisites in the master directory.

### To run:
You would need to have ARDL raw data to make use of the misc functions in data_processing.py.

To run **the eigenvalue converter training** please call the following from the master folder:
```
cd utils/pretraining_eigenvalue_converter/  
python3 eigenvalue_converter.py --niter 10001
```
It has the following arguments:
```
usage: eigenvalue_converter.py [-h] [--train_dataset_dir TRAIN_DATASET_DIR]
                               [--converter_dir CONVERTER_DIR]
                               [--num_of_nets NUM_OF_NETS]
                               [--batchSize BATCHSIZE] [--niter NITER]

optional arguments:
  -h, --help            show this help message and exit
  --train_dataset_dir TRAIN_DATASET_DIR
                        please provide the .npy arrays of training and testing
                        trajectory parameters with corresponding eigen-values
  --converter_dir CONVERTER_DIR
                        path to converter net parameters over training epochs
  --num_of_nets NUM_OF_NETS
                        number of converters to train to choose the best out
                        of
  --batchSize BATCHSIZE
                        input half-batch size
  --niter NITER         number of epochs to train for
```
...and will produce a folder with the best converter model (the one with the smallest loss on a test set), named `min_test_loss_converter.pth` and a couple of graphs with information about the training. 


