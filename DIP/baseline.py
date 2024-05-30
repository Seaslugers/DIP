import cv2
import math
import numpy as np

class Dehazer:
    def __init__(self, img, use_guided_filter=True):
        self.image = img
        self.use_guided_filter = use_guided_filter

    def dark_channel(self, img, sz):
        b, g, r = cv2.split(img)
        dc = cv2.min(cv2.min(r, g), b)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
        dark = cv2.erode(dc, kernel)
        return dark

    def atmospheric_light(self, dark):
        [h, w] = self.image.shape[:2]
        imsz = h * w
        numpx = int(max(math.floor(imsz / 1000), 1))
        darkvec = dark.reshape(imsz)
        imvec = self.image.reshape(imsz, 3)

        indices = darkvec.argsort()
        indices = indices[imsz - numpx::]

        atmsum = np.zeros([1, 3])
        for ind in range(1, numpx):
            atmsum = atmsum + imvec[indices[ind]]

        A = atmsum / numpx
        return A

    def transmission_estimate(self, A, sz):
        omega = 0.95
        im3 = np.empty(self.image.shape, self.image.dtype)

        for ind in range(0, 3):
            im3[:, :, ind] = self.image[:, :, ind] / A[0, ind]

        transmission = 1 - omega * self.dark_channel(im3, sz)
        return transmission

    def guided_filter(self, im, p, r, eps):
        mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
        mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

        q = mean_a * im + mean_b
        return q

    def transmission_refine(self, et):
        gray = (self.image * 255).astype('uint8')
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        gray = np.float64(gray) / 255
        r = 60
        eps = 0.0001
        t = self.guided_filter(gray, et, r, eps)
        return t

    def transmission_refine_no_guided(self, et):
        # 使用简单的高斯模糊代替引导滤波
        r = 15  # 核大小
        t = cv2.GaussianBlur(et, (r, r), 0)
        return t

    def recover(self, t, A, tx=0.1):
        t = cv2.max(t, tx)  # Avoid division by zero or very small values
        res = np.empty(self.image.shape, self.image.dtype)

        for ind in range(0, 3):
            res[:, :, ind] = (self.image[:, :, ind] - A[0, ind]) / t + A[0, ind]
            # Clip the result to the range [0, 1]
            res[:, :, ind] = np.clip(res[:, :, ind], 0, 1)
        return res

    def dehaze(self):
        dark = self.dark_channel(self.image, 15)
        A = self.atmospheric_light(dark)
        te = self.transmission_estimate(A, 15)
        if self.use_guided_filter:
            t = self.transmission_refine(te)
        else:
            t = self.transmission_refine_no_guided(te)
        J = self.recover(t, A, 0.1)
        return J
