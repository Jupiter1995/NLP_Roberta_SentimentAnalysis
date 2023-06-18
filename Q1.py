import pandas as pd
import numpy as np

import pyarrow.parquet as pq

from utils.transformer import SentimentCalssify
from utils.train_evaluation import TrainEvaluate


def main(model, LR, EPOCHS):
    # Open dataset for test and train

    # Open the Parquet file
    train_file = pq.ParquetFile('./train-sentiment.parquet')
    test_file = pq.ParquetFile('./test-sentiment.parquet')

    # Read the data into a Pandas DataFrame
    train_raw = train_file.read().to_pandas()
    test = test_file.read().to_pandas()

    # Split raw train dataset into train and validation datasets
    np.random.seed(112)

    # Define the proportion of data to use for validating
    val_size = 0.2

    # Calculate the number of samples to use for testing
    num_val_samples = int(len(train_raw) * val_size)

    # Generate a random permutation of the data indices
    indices = np.random.permutation(len(train_raw))

    # Split the indices into training and testing indices
    val_indices = indices[:num_val_samples]
    train_indices = indices[num_val_samples:]

    # Split the data into training and testing sets
    train_data = train_raw.iloc[train_indices]
    val_data = train_raw.iloc[val_indices]
                
    train_test = TrainEvaluate(model)
    train_test.train(train_data, val_data, LR, EPOCHS)


if __name__=="__main__":
    EPOCHS = 1
    model = SentimentCalssify(dropout=0.3)
    LR = 1e-3

    main(model, LR, EPOCHS)