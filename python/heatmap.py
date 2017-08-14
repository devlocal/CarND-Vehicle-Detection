import numpy as np
from scipy.ndimage.measurements import label


class HeatMap(object):
    """Image heat map"""

    WIDTH = 1280
    HEIGHT = 720

    # Acceptance threshold
    THRESHOLD = 3

    # Number of frames in the filter
    NUM_FRAMES = 4

    def __init__(self, aggregate_visualization=False):
        """
        :param aggregate_visualization: True to visualize aggregated data, False to visualize a single frame
        """
        self.maps = []
        self.aggregate_visualization = aggregate_visualization

    def new_frame(self):
        """
        Shifts elements in the filter, adds a new empty heat map at the end.
        """
        new_map = np.zeros(shape=(self.HEIGHT, self.WIDTH)).astype(np.uint8)

        if len(self.maps) < self.NUM_FRAMES:
            self.maps.append(new_map)
        else:
            self.maps[:-1] = self.maps[1:]
            self.maps[-1] = new_map

    def add(self, xmin, ymin, xmax, ymax):
        """
        Adds +1 for all pixels inside the box.
        """
        self.maps[-1][ymin:ymax, xmin:xmax] += 1

    def get_labels(self):
        """Returns heat map labels"""
        return label(self._get_aggregate())

    def _get_aggregate(self):
        a = np.zeros_like(self.maps[-1])
        if len(self.maps) < self.NUM_FRAMES:
            return a

        for m in self.maps:
            trs_map = m.copy()
            # Zero out pixels below the threshold
            trs_map[trs_map < self.THRESHOLD] = 0

            a = a + trs_map

        return a

    def get_visualization(self):
        """
        Returns heat map visualization.
        :return: heat map visualization
        """
        if self.aggregate_visualization:
            v = np.zeros_like(self.maps[-1])
            for m in self.maps:
                v = v + m
            return v
        else:
            return self.maps[-1]
