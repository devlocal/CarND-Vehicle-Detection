import glob
import logging
import os
import pickle
from pickle import PickleError

import cv2
import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


class Classifier(object):
    """
    Image classifier. Predicts whether an image contains a car.
    """

    VEHICLES_PATH = "../training_data/vehicles"
    NON_VEHICLES_PATH = "../training_data/non-vehicles"

    DATA_FILE = "data.p"
    MODEL_FILE = "model.p"

    VALIDATION_SPLIT = 0.2
    COLOR_TRANSFORMATION = cv2.COLOR_BGR2YCrCb

    BEST_KNOWN_HYPERPARAMS = {"C": 0.001, "dual": False}
    TRAIN_MODE = "train"

    def __init__(self, image_features):
        """
        :param image_features: list of feature classes (SpatialFeatures, ColorHistFeatures, HogFeatures).
        """
        self.path = os.path.abspath(os.path.dirname(__file__))
        self.image_features = image_features
        self.svc = None
        self.scaler = StandardScaler()

    def _load_all_files(self, relative_folder_path, label):
        data = []

        absolute_search_path = os.path.join(self.path, relative_folder_path, "**", "*.png")
        file_names = glob.glob(absolute_search_path)
        for fname in file_names:
            # Read single image
            image = cv2.imread(fname)

            # Change color space
            if self.COLOR_TRANSFORMATION:
                image = cv2.cvtColor(image, self.COLOR_TRANSFORMATION)

            # Extract features
            features = self.image_features.extract(image)
            data.append(features)

        return np.array(data, dtype=np.float32), np.ones(shape=(len(data)), dtype=np.uint8) * label

    def _load_training_data(self):
        """
        Loads training data from folders, extracts features from images.
        Pickles the dataset, subsequent program executions run faster.
        Shuffles the data set and scales features.

        :return: a 2-tuple (training data, labels)
        """
        try:
            with open(self.DATA_FILE, "rb") as f:
                X = pickle.load(f)
                y = pickle.load(f)
        except (FileNotFoundError, PickleError):
            train1, label1 = self._load_all_files(self.VEHICLES_PATH, label=1)
            train2, label2 = self._load_all_files(self.NON_VEHICLES_PATH, label=0)

            logging.info("Number of car images: %d", train1.shape[0])
            logging.info("Number of non-car images: %d", train2.shape[0])

            X = np.vstack((train1, train2))
            y = np.concatenate((label1, label2))

            with open(self.DATA_FILE, "wb") as f:
                pickle.dump(X, f)
                pickle.dump(y, f)

        X, y = sklearn.utils.shuffle(X, y)

        logging.info("Number of loaded samples: %d", y.shape[0])

        # Normalize features
        self.scaler.fit(X)
        X = self.scaler.transform(X)

        return X, y

    def _split_data(self, X, y):
        """Splits data into training and validation set"""
        rand_state = np.random.randint(0, 100)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=self.VALIDATION_SPLIT, random_state=rand_state)

        return X_train, y_train, X_valid, y_valid

    def _train(self, X, y, **hyperparams):
        """Trains the classifier with the best known set of hyperparameters"""
        X_train, y_train, X_valid, y_valid = self._split_data(X, y)

        if hyperparams:
            self.svc = LinearSVC(**hyperparams)
        self.svc.fit(X_train, y_train)

        logging.info("Training score: %.4f", self.svc.score(X_train, y_train))
        logging.info("Validation score: %.4f", self.svc.score(X_valid, y_valid))

    def _train_tune_hyperparams(self, X, y):
        """Trains the classifier, auto-tunes hyperparameters"""
        parameters = {'C': [0.1, 1, 10]}
        clf = GridSearchCV(self.svc, parameters)
        clf.fit(X, y)
        self.svc = clf.best_estimator_

        logging.info("Best params: %s", str(clf.best_params_))
        logging.info("Best score: %s", str(clf.best_score_))

    def _train_grid(self, X, y, grid):
        for hp in grid:
            print("\nTraining with {}".format(hp))
            self._train(X, y, **hp)

    def train(self):
        """
        Trains the classifier. Pickles the model, subsequent executions load trained model.
        Depending on the value of TRAIN_MODE, runs one of the following:
            "tune": runs hyperparameter auto-tuning
            "grid": trains classifier multiple times on a grid of hyperparameters
            "train": trains the classifier with hyperparameters defined by BEST_KNOWN_HYPERPARAMS
        """
        try:
            with open(self.MODEL_FILE, "rb") as f:
                self.svc = pickle.load(f)
                self.scaler = pickle.load(f)
        except (FileNotFoundError, PickleError):
            X, y = self._load_training_data()
            if self.TRAIN_MODE == "tune":
                self._train_tune_hyperparams(X, y)
            elif self.TRAIN_MODE == "grid":
                grid = [
                    {"C": 0.0002, "dual": False},
                    {"C": 0.0004, "dual": False},
                    {"C": 0.0008, "dual": False},
                    {"C": 0.001, "dual": False},
                    {"C": 0.002, "dual": False},
                    {"C": 0.004, "dual": False},
                    {"C": 0.008, "dual": False}
                ]
                self._train_grid(X, y, grid)
            elif self.TRAIN_MODE == "train":
                self._train(X, y, **self.BEST_KNOWN_HYPERPARAMS)
            else:
                raise RuntimeError("Unknown train mode {}".format(self.TRAIN_MODE))

            with open(self.MODEL_FILE, "wb") as f:
                pickle.dump(self.svc, f)
                pickle.dump(self.scaler, f)

    def predict(self, features):
        """
        Predicts whether an image container a car.

        :param features: feature vector
        :return: 1 - image contains a car, 0 - image does not contain a car
        """
        # Scale features and make a prediction
        scaled = self.scaler.transform(features)
        return self.svc.predict(scaled)

