import random

import numpy as np

from classificators.inception_time_classifier import InceptionTimeWindowClassifier
from utils.config import Config
from data_handler import DataHandler
from utils.utils import calculate_mcc_multilabel, plot_per_class_confusion

if __name__ == '__main__':

    config = Config()

    # Seeding
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    # if you use any other libraries that require seeding, set it here as well (e.g., torch.manual_seed(SEED) for PyTorch)
    # -> your results should be reproducible across runs with the same seed


    val_mccs = []
    test_mccs = []
    lr_histories_by_fold = {}

    # load data
    datahandler = DataHandler(config=config)

    # Leave-one-out: EXPERIMENT_ID = 1..4
    for fold in range(1, 5):
        print(f"\n--- Fold {fold}/4 | EXPERIMENT_ID={fold} ---")
        val_id = fold + 1 if fold < 4 else 1

        datahandler.config.data.test_experiment_id = fold
        # validation hat to be different from test
        datahandler.config.data.validation_experiment_id = val_id

        train, val, test, target_vals = datahandler.get_data_loaders()

        # just to get an insight into the data
        #plot_class_distribution(train[1], target_vals)
        #plot_class_distribution(val[1], target_vals)
        #plot_class_distribution(test[1], target_vals)

        try:

            model = InceptionTimeWindowClassifier(
                classes=target_vals,
                sensor_names=config.data.sensor_cols,
            )
            ### INSERT YOUR MODEL HERE ###
            # model = MyAwesomeModel(...)

            # Here, we use the validation set to follow good machine learning practice, which is particularly relevant for evaluation during training.
            # However, the validation data can also be incorporated directly into the model training if necessary.
            print("Training model...")
            # Note: Any kind of preprocessing, data augmentation or feature engineering should be done within the model.train() function
            # so it's capsuled within the model class (see RandomForestClassifierSK for an example)
            model.train(train, val)
            print("Evaluating model...")
            predicted_y = model.predict(test[0])

            # optional, for more insight, plot the per-class-confusion-matrix for the test set
            # plot_per_class_confusion(test[1], predicted_y, target_vals)

            # Note: The MCC might be negative for a fold since its a correlation coefficient -> Range -1 to +1
            # +1 indicates perfect correlation, 0 indicates no correlation, and -1 indicates perfect inverse correlation.
            # Since we average over classes, it can happen that some classes have a
            # negative MCC while others have a positive MCC, resulting that scores balance each other out
            # Check your individual scores to see if they are reasonable
            test_mcc = calculate_mcc_multilabel(predicted_y, test[1])
            test_mccs.append(test_mcc)

        except Exception as e:
            print(f"Fold {fold} failed with error: {e}")
            raise e
        print("--- End of Fold ---")

    avg_mcc = sum(test_mccs) / len(test_mccs)
    print("Scores for each run: ", test_mccs)
    print("\nTotal score:", avg_mcc)
