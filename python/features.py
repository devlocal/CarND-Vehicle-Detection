import cv2
import numpy as np


class ImageFeatures(object):
    """
    Composite image features.
    """

    SPATIAL_FEATURES = "spatial"
    COLOR_HIST_FEATURES = "color_hist"
    HOG_FEATURES = "hog"

    ENABLED_FEATURES = [
        SPATIAL_FEATURES,
        COLOR_HIST_FEATURES,
        HOG_FEATURES
    ]

    def __init__(self, spatial, color_hist, hog):
        """
        :param spatial: instance of SpatialFeatures or None
        :param color_hist: instance of ColorHistFeatures or None
        :param hog: instance of HogFeatures or None
        """
        self.spatial = spatial
        self.color_hist = color_hist
        self.hog = hog

    def _merge_features(self, spatial, color_hist, hog):
        features_list = []

        if self.SPATIAL_FEATURES in self.ENABLED_FEATURES:
            features_list.append(spatial)
        if self.COLOR_HIST_FEATURES in self.ENABLED_FEATURES:
            features_list.append(color_hist)
        if self.HOG_FEATURES in self.ENABLED_FEATURES:
            features_list.append(hog)

        return np.concatenate(features_list)

    @staticmethod
    def _resize_image(image, scale):
        if scale == 1:
            return image
        else:
            dest_width = round(image.shape[1] * scale)
            dest_height = round(image.shape[0] * scale)
            return cv2.resize(image, (dest_width, dest_height))

    def extract(self, image):
        """
        Extracts features from an image

        :param image: image
        :return: feature vector
        """

        spatial_features = self.spatial.extract(image)
        color_hist_features = self.color_hist.extract(image)
        hog_features = self.hog.extract(image)

        return self._merge_features(spatial_features, color_hist_features, hog_features)

    def extract_from_windows(self, image, windows):
        """
        Extracts features from image windows.

        :param image: image
        :param windows: list of windows
        :return: list of feature vectors
        """
        spatial = []
        color_hist = []
        resized_cache = {}

        # Extract spatial and color histogram features
        for xmin, ymin, xmax, ymax, scale in windows:
            try:
                resized = resized_cache[scale]
            except KeyError:
                resized = self._resize_image(image, scale)
                resized_cache[scale] = resized

            subimg = resized[ymin:ymax, xmin:xmax]

            s = self.spatial.extract(subimg)
            ch = self.color_hist.extract(subimg)

            spatial.append(s)
            color_hist.append(ch)

        # Extract HOG features
        hog = self.hog.extract_from_windows(resized_cache, windows)

        result = []
        for f in zip(spatial, color_hist, hog):
            result.append(self._merge_features(*f))
        return result
