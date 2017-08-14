from contextlib import suppress

import numpy as np
from skimage.feature import hog


class HogFeatures(object):
    """
    HOG features
    """

    HOG_CHANNEL = "ALL"
    PIX_PER_CELL = 11
    CELL_PER_BLOCK = 2
    ORIENT = 12

    IMAGE_WIDTH = 1280
    BASE_LENGTH = 64
    NBLOCKS_PER_WINDOW = (BASE_LENGTH // PIX_PER_CELL) - CELL_PER_BLOCK + 1

    def __init__(self):
        self._search_range = {}

    def _get_features(self, channel, feature_vec=True):
        """
        Computs HOG features.

        :param channel: image channel
        :param feature_vec: True to return feature vector, otherwise returns a 2-tuple
        :return: either a 2-tuple (features, hog_images) or features
        """
        return hog(
            channel,
            orientations=self.ORIENT,
            pixels_per_cell=(self.PIX_PER_CELL, self.PIX_PER_CELL),
            cells_per_block=(self.CELL_PER_BLOCK, self.CELL_PER_BLOCK),
            transform_sqrt=True,
            visualise=False,
            feature_vector=feature_vec
        )

    def _get_channel_features(self, subimage):
        """Computes individual channel HOG features for the entire image"""
        hog1 = self._get_features(subimage[:, :, 0], feature_vec=False) if self.HOG_CHANNEL in ["ALL", 0] else None
        hog2 = self._get_features(subimage[:, :, 1], feature_vec=False) if self.HOG_CHANNEL in ["ALL", 1] else None
        hog3 = self._get_features(subimage[:, :, 2], feature_vec=False) if self.HOG_CHANNEL in ["ALL", 2] else None

        return hog1, hog2, hog3

    def _get_search_range(self, windows, scale):
        try:
            # Assumes that windows do not change from iteration to iteration.
            return self._search_range[scale]
        except KeyError:
            y_start = min([w[1] for w in windows if w[4] == scale])
            y_stop = max([w[3] for w in windows if w[4] == scale])

            self._search_range[scale] = y_start, y_stop
            return self._search_range[scale]

    def extract(self, image):
        """
        Extracts HOG features from image. If configured to extract features from all channels,
        consolidates all features into a single vector.

        :param image: image
        :return: features
        """
        if self.HOG_CHANNEL == 'ALL':
            hog_features = []
            for channel in range(image.shape[2]):
                hog_features.append(self._get_features(image[:, :, channel], feature_vec=True))
            features = np.ravel(hog_features)
        else:
            features = self._get_features(image[:, :, self.HOG_CHANNEL], feature_vec=True)

        return features

    def extract_from_windows(self, resized_cache, windows):
        """
        Extracts features from image windows.

        :param resized_cache: cache of resized images
        :param windows: windows
        :return: list of features
        """
        hog_cache = {}

        result = []
        for xmin, ymin, xmax, ymax, scale in windows:
            y_start, y_stop = self._get_search_range(windows, scale)
            img = resized_cache[scale]
            img_tosearch = img[y_start:y_stop, :, :]

            try:
                hog1, hog2, hog3 = hog_cache[scale]
            except KeyError:
                hog1, hog2, hog3 = self._get_channel_features(img_tosearch)
                hog_cache[scale] = hog1, hog2, hog3

            assert ymax - ymin == self.BASE_LENGTH
            assert xmax - xmin == self.BASE_LENGTH

            xpos = int(xmin // self.PIX_PER_CELL)
            ypos = int((ymin - y_start) // self.PIX_PER_CELL)

            # Extract HOG for this patch
            features = []

            with suppress(TypeError):
                hog_feat1 = hog1[ypos:ypos + self.NBLOCKS_PER_WINDOW, xpos:xpos + self.NBLOCKS_PER_WINDOW].ravel()
                features.append(hog_feat1)

            with suppress(TypeError):
                hog_feat2 = hog2[ypos:ypos + self.NBLOCKS_PER_WINDOW, xpos:xpos + self.NBLOCKS_PER_WINDOW].ravel()
                features.append(hog_feat2)

            with suppress(TypeError):
                hog_feat3 = hog3[ypos:ypos + self.NBLOCKS_PER_WINDOW, xpos:xpos + self.NBLOCKS_PER_WINDOW].ravel()
                features.append(hog_feat3)

            hog_features = np.hstack(features)

            result.append(hog_features)

        return result
