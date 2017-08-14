import cv2
import numpy as np


class LabelsVisualization(object):
    """
    Heatmap labels visualization
    """

    @staticmethod
    def draw(image, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(image, bbox[0], bbox[1], (0, 0, 255), 6)
        # Return the image
        return image


class WindowsVisualization(object):
    """
    Sliding windows visualization
    """

    @staticmethod
    def draw(image, windows):
        for x1, y1, x2, y2, scale in windows:
            top_left = (round(x1 / scale), round(y1 / scale))
            bottom_right = (round(x2 / scale), round(y2 / scale))
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 255), 1)
        return image


class HeatmapVisualization(object):
    """
    Heatmap visualization
    """

    TRANSPARENCY = 1
    WEIGHT = 40

    def draw(self, image, heatmap):
        z = np.zeros(shape=image.shape[:2])
        r = np.array(heatmap.get_visualization() * self.WEIGHT)
        r = np.clip(r, 0, 255)
        mask = np.dstack((r, z, z)).astype(np.uint8)
        return cv2.addWeighted(image, 1, mask, self.TRANSPARENCY, 0)
