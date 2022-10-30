from typing import Tuple, Union
from os.path import exists
from os import makedirs
import cv2
import numpy as np
from stereo.settings import pattern_size, path


class Capture():
    def __init__(self) -> None:
        self._cap_left = cv2.VideoCapture(0)
        self._cap_right = cv2.VideoCapture(1)
        self._window_result = cv2.namedWindow("capture", cv2.WINDOW_AUTOSIZE)
        if not exists(path):
            makedirs(path)

    def _detect_pattern(self, img: np.array) -> Tuple[bool, Union[np.array, None]]:
        found, corners = cv2.findChessboardCorners(img, pattern_size)
        if found:
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(vis, pattern_size, corners, found)
            return True, vis
        return False, None

    def capture_image(self, nums: int = 20) -> bool:
        i = 0
        while i < nums:
            ret0, frame_left = self._cap_left.read()
            ret1, frame_right = self._cap_right.read()
            if ret0 and ret1:
                frame_left = cv2.resize(
                    frame_left, (640, 480), interpolation=cv2.CV_8SC1)
                frame_right = cv2.resize(
                    frame_right, (640, 480), interpolation=cv2.CV_8SC1)
                frame = np.concatenate([frame_left, frame_right], axis=1)

                found1, vis1 = self._detect_pattern(
                    cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY))
                found2, vis2 = self._detect_pattern(
                    cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY))
                if found1 and found2:
                    image = np.concatenate([vis1, vis2], axis=1)
                    frame = np.concatenate([frame, image], axis=0)
                else:
                    frame = np.concatenate(
                        [frame, frame], axis=0)

                cv2.imshow("capture", frame)
                key = cv2.waitKey(30)
                if key == ord("s") and found1 and found2:
                    i = i + 1
                    print(f"{i} images captured")
                    cv2.imwrite((f'{path}{i}left.jpg'), frame_left)
                    cv2.imwrite((f'{path}{i}right.jpg'), frame_right)
                if key & 0xFF == ord('q'):
                    break
            else:
                raise Exception("Capture failed!")
