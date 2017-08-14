import numpy as np


class ColorHistFeatures(object):
    """
    Color histogram feature extractor
    """

    # Number of histogram bins
    HIST_BINS = 32

    # Range of feature values
    BINS_RANGE = (0, 256)

    def extract(self, image):
        """
        Extracts color histogram features from an image

        :param image: image
        :return: ndarray with features extracted from each color channel separately
        """

        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(image[:, :, 0], bins=self.HIST_BINS, range=self.BINS_RANGE)
        channel2_hist = np.histogram(image[:, :, 1], bins=self.HIST_BINS, range=self.BINS_RANGE)
        channel3_hist = np.histogram(image[:, :, 2], bins=self.HIST_BINS, range=self.BINS_RANGE)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features
