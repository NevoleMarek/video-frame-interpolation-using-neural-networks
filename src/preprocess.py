import os

import cv2 as cv
import numpy as np

from config import PARTITION_SIZE, PARTITIONS_PATH, VIDEOS_PATH, SAMPLE_RATE


class Video:
    def __init__(self, video_capture):
        self.cap = video_capture

    @classmethod
    def from_file(cls, filename):
        cap = cv.VideoCapture(filename)
        return cls(cap)

    def get_frame(self):
        return self.cap.read()

    def get_resized_frame(self, dim = (256,144)):
        ret, frame = self.get_frame()
        if ret:
            frame = cv.resize(
                frame,
                dim,
                interpolation = cv.INTER_LANCZOS4
            )
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        return ret, frame

    def extract_cnn_sample(self, dim = (256,144)):
        i = 0
        while True:
            if i != SAMPLE_RATE:
                i += 1
                ret, img = self.get_frame()
                if not ret:
                    break
                else:
                    continue
            else:
                i=0
                ret_1, img_1 = self.get_resized_frame(dim)
                ret_2, img_2 = self.get_resized_frame(dim)
                ret_3, img_3 = self.get_resized_frame(dim)
                if not all([ret_1, ret_2, ret_3]):
                    break
                else:
                    yield np.concatenate((img_1, img_3), axis=2), img_2

    def release(self):
        self.cap.release()


def main():
    videos = os.listdir(VIDEOS_PATH)
    size = 0
    partition_idx = 0
    partition_X = []
    partition_Y = []
    for video in videos:
        v = Video.from_file(os.path.join(VIDEOS_PATH,video))
        for x, y in v.extract_cnn_sample():
            partition_X.append(x)
            partition_Y.append(y)
            size += 1
            if size == PARTITION_SIZE:
                path = os.path.join(PARTITIONS_PATH,f'partition_{partition_idx:04}.npz')
                X = np.stack(partition_X)
                Y = np.stack(partition_Y)
                np.savez_compressed(path ,X=X, Y=Y)
                print(f'Partition_{partition_idx:04}')
                size = 0
                partition_X = []
                partition_Y = []
                partition_idx += 1

        v.release()
        print(f'Done video {video}')
    if size != 0:
        path = os.path.join(PARTITIONS_PATH,f'partition_{partition_idx:04}.npz')
        X = np.stack(partition_X)
        Y = np.stack(partition_Y)
        np.savez_compressed(path ,X=X, Y=Y)


if __name__ == '__main__':
    main()
