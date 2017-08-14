import os

import cv2
from moviepy.editor import VideoFileClip


class VideoProcessor(object):
    """Frame-by-frame video processor"""

    COLOR_TRANSFORMATION = cv2.COLOR_RGB2YCrCb

    def __init__(self, frame_handler):
        """
        :param frame_handler: frame handler to apply to each single frame
        """
        self._frame_handler = frame_handler

    def _image_func(self, frame):
        if self.COLOR_TRANSFORMATION:
            converted = cv2.cvtColor(frame, self.COLOR_TRANSFORMATION)
        else:
            converted = frame
        return self._frame_handler(frame, converted)

    def process_video(self, file_name, out_file_name=None, trim_range_seconds=None):
        """Processes video clip with the help of frame handler"""

        if not out_file_name:
            out_file_name = "{}-out{}".format(*os.path.splitext(file_name))

        clip = VideoFileClip(file_name)
        if trim_range_seconds:
            clip = clip.subclip(*trim_range_seconds)
        out_clip = clip.fl_image(self._image_func)
        out_clip.write_videofile(out_file_name, audio=False)
