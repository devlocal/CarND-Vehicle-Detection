class VehicleTracker(object):
    """
    VehicleTracker processes each video frame and marks areas with cars
    """
    def __init__(self, sliding_window, labels_visualization,
                 windows_visualization=None, heatmap_visualization=None):
        self.sliding_window = sliding_window
        self.labels_visualization = labels_visualization
        self.windows_visualization = windows_visualization
        self.heatmap_visualization = heatmap_visualization

    def _visualize(self, visualization_frame, heatmap):
        """Visualises image analysis results"""
        if self.windows_visualization:
            visualization_frame = self.windows_visualization.draw(visualization_frame, self.sliding_window.windows)
        if self.heatmap_visualization:
            visualization_frame = self.heatmap_visualization.draw(visualization_frame, heatmap)
        visualization_frame = self.labels_visualization.draw(visualization_frame, heatmap.get_labels())
        return visualization_frame

    def process_frame(self, visualization_frame, frame):
        """
        Processes a single video frame. Takes two images to allow visualization frame to have a different color space.

        :param visualization_frame: frame to draw visualization on
        :param frame: from for image analysis
        :return: visualization frame
        """

        heatmap = self.sliding_window.search(frame)
        return self._visualize(visualization_frame, heatmap)
