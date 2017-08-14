import cv2


class SpatialFeatures(object):
    """Spatial features"""

    SPATIAL_SIZE = (32, 32)

    def extract(self, image):
        """
        Extracts spatial features from image.

        :param image: image
        :return: features
        """
        return cv2.resize(image, self.SPATIAL_SIZE).ravel()
