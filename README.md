# Household Electricity Consumption Forecasting using LSTM

## 1. Project Overview

This project predicts household electricity consumption using a Long Short-Term Memory (LSTM) neural network.

The model learns patterns from historical electricity measurements and predicts the next minute's global active power consumption using the previous 60 minutes of measurements.

**Objective:** Forecast electricity consumption to support energy monitoring, demand estimation, and electricity distribution planning.

## 2. Problem Statement

An electricity provider needs to forecast household electricity consumption for efficient energy distribution.

The project uses 10,000 consecutive minute-level records from the Household Electric Power Consumption dataset. Missing values are handled, numerical features are normalized, and historical sequences are generated to train a recurrent neural network.

The trained model is evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). Actual and predicted consumption are also visualized and tested on unseen records.

## 3. Dataset

**Dataset name:** Household Electric Power Consumption

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

The dataset contains measurements of household electric power consumption collected at one-minute intervals.

### Features used

| Feature | Description |
|---|---|
| Global_active_power | Total active power consumed by the household, in kW |
| Global_reactive_power | Reactive power, in kW |
| Voltage | Average voltage, in volts |
| Global_intensity | Current intensity, in amperes |
| Sub_metering_1 | Energy usage by sub-meter 1, in Wh |
| Sub_metering_2 | Energy usage by sub-meter 2, in Wh |
| Sub_metering_3 | Energy usage by sub-meter 3, in Wh |

The original dataset also contains Date and Time columns. These are combined into a DateTime column to preserve chronological ordering.

**Selected data:** The first 10,000 consecutive records from the dataset.

## 4. Technologies Used

- Python — programming language
- Google Colab — cloud-based notebook execution
- Pandas — dataset loading and preprocessing
- NumPy — numerical operations and sequence generation
- Scikit-learn — MinMaxScaler and evaluation metrics
- TensorFlow / Keras — LSTM model construction and training
- Matplotlib — visualization of actual and predicted consumption

## 5. Why LSTM?

LSTM stands for Long Short-Term Memory. It is a type of recurrent neural network designed to learn patterns from sequential data.

Electricity consumption changes over time. Recent measurements can help predict the next measurement, so a model that learns temporal dependencies is suitable for this problem.

LSTM uses memory cells and gates to control which information is retained or forgotten. This helps it learn relationships across a sequence of previous measurements.

### Why not other models?

- **Simple RNN:** Can learn sequential patterns, but may struggle with long-term dependencies because of vanishing gradients.
- **GRU:** A strong alternative with fewer gates and often fewer parameters. It was not tested in this experiment, so its performance cannot be directly compared with LSTM here.
- **CNN:** Can learn local patterns from sequences, but a CNN model was not implemented or evaluated in this experiment.
- **Dense neural network:** Can learn nonlinear relationships, but does not inherently process temporal sequences in the same way as a recurrent model.
- **Transformer:** Can model long-range relationships, but may require more computational resources and is not necessary for establishing this baseline.

**Model selection:** LSTM was selected because the task involves sequential time-series forecasting and LSTM is designed to learn temporal dependencies. Alternative models could be tested in future experiments.

## 6. Data Preprocessing

### Step 1: Load the dataset

The uploaded ZIP file contains `household_power_consumption.txt`. Pandas reads the file using a semicolon delimiter and treats `?` as a missing value.

Only the first 10,000 records are loaded to satisfy the experiment's data-size requirement.

### Step 2: Handle missing values

Linear interpolation estimates missing measurements from surrounding values. Forward fill and backward fill are used as fallbacks.

The dataset contains zero missing values after preprocessing.

### Step 3: Convert Date and Time

The Date and Time columns are combined into DateTime and sorted chronologically. The interval check confirms that the selected records are one minute apart.

### Step 4: Normalize numerical features

MinMaxScaler scales the seven numerical features using the training portion's minimum and maximum values.

The scaler is fitted only on the first 80% of the chronological records to reduce test-data leakage. Test values may exceed the range 0–1 if they fall outside the training range.

### Step 5: Generate sequences

The model uses a 60-minute historical window to predict the next minute's Global_active_power.

Each input sequence has 60 time steps and seven features.

- Total records: 10,000
- Training sequences: 7,940
- Test sequences: 2,000
- Input shape: (60, 7)
- Prediction target: next-minute Global_active_power

The data is split chronologically, using the first 80% for training and the final 20% for testing. The test sequences retain the preceding 60 minutes as historical input.

## 7. LSTM Model Architecture

The model contains the following layers:

| Layer | Configuration | Purpose |
|---|---|---|
| Input | 60 time steps × 7 features | Receives historical measurements |
| LSTM 1 | 32 units | Learns temporal patterns |
| Dropout | 0.2 | Randomly drops 20% of activations during training to help reduce overfitting |
| LSTM 2 | 16 units | Learns higher-level sequential representations |
| Dense | 1 output | Predicts next-minute active power |

**Total trainable parameters:** 8,273

### Why two LSTM layers?

The first LSTM layer returns a sequence so that the second LSTM layer can process the representations from all 60 time steps. The second LSTM produces a representation used by the final Dense layer.

### Why Dense(1)?

Only one numerical value is predicted: the next minute's Global_active_power.

## 8. Hyperparameters

Hyperparameters are configuration values chosen before or during model development. They control model architecture and training behavior.

| Hyperparameter | Value used | Explanation |
|---|---|---|
| Epochs | 10 | Number of complete passes through the training data |
| Batch size | 64 | Number of sequences processed per training batch |
| Validation split | 0.1 (10%) | Fraction of training data reserved for validation |
| Optimizer | Adam | Updates model weights to minimize the loss |
| Loss function | Mean Squared Error (MSE) | Penalizes squared prediction errors during training |
| First LSTM units | 32 | Number of units in the first LSTM layer |
| Second LSTM units | 16 | Number of units in the second LSTM layer |
| Dropout | 0.2 | Dropout rate of 20% |
| Sequence length | 60 | Number of previous minutes used for each prediction |
| Train/test split | 80% / 20% | Chronological split for model evaluation |
| Random seed | 42 | Sets TensorFlow's random seed for improved reproducibility |

### What is an epoch?

One epoch means the model has processed the entire training dataset once. This experiment uses 10 epochs, so the model trains for 10 passes over the training sequences.

More epochs do not automatically guarantee better performance. Too many epochs may lead to overfitting, while too few may result in underfitting.

### What is batch size?

Batch size determines how many training sequences are processed before a weight update. With a batch size of 64, the model processes training samples in groups of up to 64 sequences.

### How can hyperparameters be changed?

- Increase epochs to 15 or 20 to test whether the model improves.
- Change batch size to 32 or 128 to compare training speed and performance.
- Change the LSTM units to 64 and 32 to test a larger model.
- Change sequence length from 60 to 30 or 120 to compare different historical windows.
- Change dropout from 0.2 to 0.3 if overfitting is observed.

These are possible future experiments, not results measured in the current run. Compare configurations using validation performance and evaluate the final selected model on the held-out test set.

## 9. Model Training

The model is compiled with the Adam optimizer and Mean Squared Error loss.

It is trained using the training sequences for 10 epochs with a batch size of 64. Ten percent of the training data is used for validation, and shuffling is disabled to preserve temporal ordering.

The training loss and validation loss decrease over the 10 epochs, indicating that the model is learning patterns in the selected dataset. Lower training loss alone does not prove good generalization; held-out test metrics are also required.

## 10. Evaluation Metrics

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted values.

MAE = (1/n) × Σ |actual − predicted|

A lower MAE indicates that predictions are closer to actual measurements on average.

### Root Mean Squared Error (RMSE)

RMSE is the square root of the average squared prediction error.

RMSE = √[(1/n) × Σ (actual − predicted)²]

RMSE penalizes larger errors more heavily than MAE.

Both metrics are calculated after converting predictions back to the original Global_active_power scale.

## 11. Experimental Results

Results from the current Colab experiment:

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 0.5742 kW |
| Root Mean Squared Error (RMSE) | 0.7686 kW |

The actual-versus-predicted graph compares the first 300 test predictions with their corresponding actual measurements.

The model follows several consumption trends but does not capture every sudden spike accurately. The MAE and RMSE provide quantitative measures of the remaining prediction error.

These results represent the current experimental run and should not be interpreted as a guarantee of future forecasting accuracy.

## 12. Unseen Test Predictions

The final 20% of the chronological dataset is reserved for testing and is not used for fitting the model weights.

The notebook displays the first 10 actual and predicted values from the test set. These examples demonstrate how the trained model performs on held-out records.

This is a held-out test demonstration using records from the same dataset, not a test on a separate household or an independently collected dataset.

## 13. How to Run the Project

1. Open the notebook in Google Colab.
2. Upload the dataset ZIP file when requested by the notebook.
3. Run the cells in order, from data loading through final evaluation.
4. Wait for model training to finish.
5. Review the MAE, RMSE, actual-versus-predicted graph, and unseen test predictions.

The notebook requires internet access when downloading or uploading the dataset, but the loaded data can be processed in the active Colab runtime.

## 14. Limitations

- The experiment uses only 10,000 records rather than the entire dataset.
- The first 10,000 records cover a relatively short period and may not represent daily or seasonal patterns.
- The model predicts only one minute ahead.
- Linear interpolation can introduce estimated values where measurements are missing.
- Sudden consumption spikes can be difficult to predict accurately.
- Alternative architectures and hyperparameter configurations have not been benchmarked in this experiment.

## 15. Future Improvements

- Train and evaluate on a larger contiguous dataset.
- Compare LSTM, GRU, simple RNN, and CNN-based forecasting models.
- Tune sequence length, batch size, dropout, and number of LSTM units.
- Use early stopping and model checkpoints.
- Compare against simple persistence and statistical forecasting baselines.
- Evaluate longer forecasting horizons and additional calendar features.
- Test on a separate time period or another household where suitable data is available.

## 16. Conclusion

This project demonstrates a complete time-series forecasting workflow using an LSTM neural network: loading data, handling missing values, normalizing features, generating historical sequences, training a recurrent model, evaluating prediction errors, and visualizing results.

The experiment provides a baseline for predicting household electricity consumption and offers several opportunities for further model improvement.

## Author

GitHub: [24cs183-blip](https://github.com/24cs183-blip)

**Project:** Household Electricity Consumption Forecasting using LSTM
