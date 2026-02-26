# Pal2Sim CPS IoT Activity Recognition Challenge 2026
## Welcome to the Pal2Sim CPS IoT Activity Recognition Challenge 2026! 

The goal of this challenge is to classify industrial vehicle activities (e.g., driving, lifting, idling) based on 
multi-sensor time-series data. We provide a robust dataset, a baseline model, and a standardized evaluation framework.

## 🏆 The Task
You are given time-series data from 4 different experiments (Train & Validation). 
Your task is to train a model that generalizes well to a 4th unseen experiment (Test).

Input: 3-Dimensional Sensor Data (Samples, Time_Steps, Sensors)

Output: Activity Class (Multi-class / Multi-label)

Metric: Matthews Correlation Coefficient (MCC)

## 🚀 Getting Started
1. Installation
Clone the repository and install the dependencies. We recommend using a virtual environment (Python 3.12+).

```bash
git clone
cd pal2sim-challenge
pip install -r requirements.txt
```

2. Download Data 
With your registration you should have received access to the dataset. Paste the download link in the config file (config.py) under the variable `DATA_URL`.
You don't need to manually download the dataset. The code handles this for you. Just run the baseline script once, and it will fetch the data to ./data/.

3. Run the Baseline
We provide a Dummy and Random Forest Baseline that uses statistical feature extraction (Mean, Std, Min, Max).
Those are just for your reference. You are encouraged to build your own models and replacing the baseline code.

```bash
python main.py
```


## 🛠️ Evaluation Metric
We use the Matthews Correlation Coefficient (MCC) because the classes are highly imbalanced.

+1.0: Perfect prediction.
0.0: Random guessing.
-1.0: Total disagreement.

The final evaluation is performed on the hidden Test Set.
Each experiment is treated as a separate test fold. The final score is the average MCC across all folds.

Please note that your task is limited to the model. 
You can make changes to the evaluation locally (e.g., by building hyperparameter tuning around it), 
but the final evaluation will be based on the method defined in this repo! 

## 📝 Rules & Guidelines

See [Rulebook](https://www.pal2sim.com/rulebook.pdf) for detailed rules and guidelines.

## General Information
All information can be found on the [Pal2Sim Competition Website](https://www.pal2sim.com/?tab=competition).


## Contact 
For questions, please reach out to us at [Pal2Sim Competition](mailto:pal2sim-competition@iml.fraunhofer.de)
