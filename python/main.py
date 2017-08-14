#!/usr/bin/env python

import os

from classifier import Classifier
from features import ImageFeatures
from heatmap import HeatMap
from sliding_window import SlidingWindowSearch
from tracker import VehicleTracker
from basic_logging import setup_basic_logging
from video import VideoProcessor

from color_hist import ColorHistFeatures
from hog import HogFeatures
from spatial import SpatialFeatures
from visialization import LabelsVisualization, WindowsVisualization, HeatmapVisualization

INPUT_VIDEO_FILES = [
    "project_video.mp4",
    "test_video.mp4",
]


def create_image_features():
    """Creates ImageFeatures instance"""
    spatial = SpatialFeatures()
    color_hist = ColorHistFeatures()
    hog = HogFeatures()

    return ImageFeatures(spatial, color_hist, hog)


def create_classifier(image_features):
    """Creates a classifier instance"""
    classifier = Classifier(image_features)
    classifier.train()
    return classifier


def process_video(input_file_name, output_file_name, sliding_window, labels_visualization,
                  windows_visualization=None, heatmap_visualization=None, trim_range_seconds=None):
    """Processes video clip"""

    tracker = VehicleTracker(sliding_window, labels_visualization, windows_visualization, heatmap_visualization)

    # Process video file
    processor = VideoProcessor(tracker.process_frame)
    processor.process_video(
        file_name=input_file_name,
        out_file_name=output_file_name,
        trim_range_seconds=trim_range_seconds
    )


def main():
    setup_basic_logging()

    debug_kwargs = {
        "trim_range_seconds": (40, 42),
        "windows_visualization": WindowsVisualization(),
        "heatmap_visualization": HeatmapVisualization(),
    }

    image_features = create_image_features()
    classifier = create_classifier(image_features)
    heatmap = HeatMap()
    sliding_window = SlidingWindowSearch(image_features, classifier, heatmap)
    labels_visualization = LabelsVisualization()

    process_video(
        input_file_name=os.path.join("..", INPUT_VIDEO_FILES[0]),
        output_file_name="../temp/debug.mp4",
        sliding_window=sliding_window,
        labels_visualization=labels_visualization,
        # **debug_kwargs
    )


if __name__ == "__main__":
    main()
