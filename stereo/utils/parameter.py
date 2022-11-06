import cv2
from stereo.settings import size


class _Parameters:
    def __init__(self, fs_) -> None:
        self.fs = cv2.FileStorage(fs_, cv2.FILE_STORAGE_READ)


class _Intrinsics(_Parameters):
    def __init__(self, fs_) -> None:
        super().__init__(fs_)
        self.M1 = self.fs.getNode("M1").mat()
        self.D1 = self.fs.getNode("D1").mat()
        self.M2 = self.fs.getNode("M2").mat()
        self.D2 = self.fs.getNode("D2").mat()


class _Extrinsic(_Parameters):
    def __init__(self, fs_) -> None:
        super().__init__(fs_)
        self.R = self.fs.getNode("R").mat()
        self.T = self.fs.getNode("T").mat()
        self.R1 = self.fs.getNode("R1").mat()
        self.R2 = self.fs.getNode("R2").mat()
        self.P1 = self.fs.getNode("P1").mat()
        self.P2 = self.fs.getNode("P2").mat()
        self.Q = self.fs.getNode("Q").mat()


class StereoParameters:
    def __init__(self, intrinsics, extrinsics) -> None:
        self.intrinsics = _Intrinsics(intrinsics)
        self.extrinsics = _Extrinsic(extrinsics)

    def get_rectify_map_left(self):
        return cv2.initUndistortRectifyMap(self.intrinsics.M1,
                                           self.intrinsics.D1,
                                           self.extrinsics.R1,
                                           self.extrinsics.P1,
                                           size,
                                           cv2.CV_32FC2)
    
    def get_rectify_map_right(self):
        return cv2.initUndistortRectifyMap(self.intrinsics.M2,
                                           self.intrinsics.D2,
                                           self.extrinsics.R2,
                                           self.extrinsics.P2,
                                           size,
                                           cv2.CV_32FC2)
