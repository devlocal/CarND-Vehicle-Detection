class SlidingWindowSearch(object):
    """
    Sliding window search implementation
    """

    IMAGE_WIDTH = 1280

    # Sliding window size. Instead of changing window size, image is resized. Window size remains constant.
    BASE_LENGTH = 64

    # Search windows
    #  n_windows -- number of windows in a single row from the left edge to the right edge
    #  overlap -- fraction of overlap
    #  y_start -- top of search area
    #  y_stop -- bottom of search area
    WINDOWS = [
        {"n_windows": 20, "overlap": 0.25, "y_start": 400, "y_stop": 520},
        {"n_windows": 17, "overlap": 0.25, "y_start": 390, "y_stop": 520},
        {"n_windows": 15, "overlap": 0.25, "y_start": 390, "y_stop": 560},
        {"n_windows": 13, "overlap": 0.25, "y_start": 390, "y_stop": 600},
    ]

    def __init__(self, features, classifier, heatmap):
        self.features = features
        self.classifier = classifier
        self.heatmap = heatmap
        self.windows = self._build_window_list()

    def _build_window_list(self):
        """
        :return: a list
          [
              (x1, y1, x2, y2, dest_width),
              (x1, y1, x2, y2, dest_width),
              ...
          ]
        """
        window_list = []

        for wdef in self.WINDOWS:
            n_windows = wdef["n_windows"]
            overlap = wdef["overlap"]
            ymin = wdef["y_start"]
            ymax = wdef["y_stop"]

            dest_width = n_windows * self.BASE_LENGTH
            assert dest_width <= self.IMAGE_WIDTH, "Invalid dest with {} for '{}'".format(dest_width, wdef)
            scale = float(dest_width) / self.IMAGE_WIDTH

            ymin = round(ymin * scale)
            ymax = round(ymax * scale)
            xmin = 0
            xmax = dest_width

            fx = 0.0
            x = xmin
            while x + self.BASE_LENGTH <= xmax:
                for y in range(ymin, ymax - self.BASE_LENGTH + 1, round(self.BASE_LENGTH * overlap)):
                    window_list.append((x, y, x + self.BASE_LENGTH, y + self.BASE_LENGTH, scale))
                fx += overlap
                x = round(self.BASE_LENGTH * fx)
        return window_list

    def search(self, image):
        """
        Runs a search using provided feature class to extract features and classifier to classify each window.
        :param image: image
        :return: heatmap
        """
        self.heatmap.new_frame()

        features = self.features.extract_from_windows(image, self.windows)
        for (x1, y1, x2, y2, scale), f in zip(self.windows, features):
            prediction = self.classifier.predict(f.reshape(1, -1))
            if prediction == 1:
                self.heatmap.add(round(x1 / scale), round(y1 / scale), round(x2 / scale), round(y2 / scale))

        return self.heatmap
