import numpy as np
import scipy.misc as misc
import cv2
import matplotlib.pyplot as pplot

def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the 0idth and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)

class Gist:
    def __init__(self, image_size=128,orientations=(8, 8, 8), num_blocks=4):    
    
        self.image_size = image_size
        self.orientations = orientations
        self.num_blocks = num_blocks
        self.gabors = self._create_gabor(self.orientations, self.image_size)

   
    def _prefilter(self,img, fc=4, w=5):
        s1 = fc / np.sqrt(np.log(2))

        # Pad images to reduce boundary artifacts
        img = np.log(img+1)
        img = np.lib.pad(img, ((w, w), (w, w)), 'symmetric')
        sn, sm = img.shape
        n = max((sn, sm))
        n = n + n % 2
        img = np.lib.pad(img, ((0, n-sn), (0, n-sm)), 'symmetric')

        # Filter
        fx, fy = np.meshgrid(np.arange(-n/2, n/2), np.arange(-n/2, n/2))
        gf = np.fft.fftshift(np.exp(-(fx ** 2 + fy ** 2) / (s1 ** 2)))

        # Whitening
        output = img - np.real(np.fft.ifft2(np.multiply(np.fft.fft2(img), gf)))

        # Local contrast normalization
        tmp = np.fft.ifft2(np.multiply(np.fft.fft2(output ** 2), gf))
        localstd = np.sqrt(np.abs(tmp))
        output /= (0.2 + localstd)

        # Crop output to have same size than the input
        output = output[w:sn-w, w:sm-w]
        return output

   
    def _get_feature(self,img, w, G):
        '''Estimate global features.'''

        def average(x, N):
            '''Average over non-overlapping square image blocks.'''
            nx = np.fix(np.linspace(0, x.shape[0], N+1)).astype(np.int16)
            ny = np.fix(np.linspace(0, x.shape[1], N+1)).astype(np.int16)
            y = np.zeros((N, N))
            for xx in range(N):
                for yy in range(N):
                    v = np.mean(x[nx[xx]:nx[xx+1], ny[yy]:ny[yy+1]])
                    y[yy, xx] = v
            return y

        n, n, num_filters = G.shape
        W = w * w
        g = np.zeros((W * num_filters, 1))

        img = np.fft.fft2(img)
        k = 0
        for n in range(num_filters):
            ig = np.abs(np.fft.ifft2(np.multiply(img, G[:, :, n])))
            v = average(ig, w)
            g[k:k+W, :] = np.reshape(v, (W, 1))
            k += W
        return g

    def _create_gabor(self,orientations, n):
        '''Compute filter transfer functions.'''

        Nscales = len(orientations)
        num_filters = sum(orientations)

        param = []
        for i in range(Nscales):
            for j in range(orientations[i]):
                param.append([.35,
                              .3/(1.85 ** i),
                              16.0 * orientations[i] ** 2 / 32 ** 2,
                              np.pi / orientations[i] * j])
        param = np.array(param)

        # Frequencies:
        fx, fy = np.meshgrid(np.arange(-n/2, n/2), np.arange(-n/2, n/2))
        fr = np.fft.fftshift(np.sqrt(fx ** 2 + fy ** 2))
        t = np.fft.fftshift(np.angle(fx + 1j * fy))

        # Transfer functions:
        gabors = np.zeros((n, n, num_filters))
        for i in range(num_filters):
            tr = t + param[i, 3]
            tr = tr + 2 * np.pi * (tr < -np.pi) - 2 * np.pi * (tr > np.pi)
            gabors[:, :, i] = np.exp(
                - 10 * param[i, 0] * (fr / n / param[i, 1] - 1) ** 2
                - 2 * param[i, 2] * np.pi * tr ** 2)
        return gabors

    def get_gist_features(self,img):
        img = cv2.resize(img, (self.image_size, self.image_size))    
        output = self._prefilter(img.astype(np.float16))
        features = self._get_feature(output, self.num_blocks, self.gabors)
        return np.squeeze(features.flatten())


def read_video():
    file_path = "test.mp4"
    cap = cv2.VideoCapture(file_path)
    fps = 0;
    capture_every = 30
    while not cap.isOpened():
        cap = cv2.VideoCapture(file_path)                

            
    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    while True:
        flag, frame = cap.read()
        frame = maintain_aspect_ratio_resize(frame, width=320)
        if flag:
                pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if pos_frame%capture_every == 0 :
                    # The frame is ready and already captured
                    cv2.imshow('video', frame)
                
                    print (str(pos_frame)+" frames")
                    cv2.waitKey(100)
                    yield frame            
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
            print ("frame is not ready")
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)
    
        if cv2.waitKey(10) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break

    
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    reader = read_video()
    gist = Gist(image_size=128)
    import time
    for frame in reader:
        ts = time.time()    
        g0 = gist.get_gist_features(frame[:,:,0])
        g1 = gist.get_gist_features(frame[:,:,1])
        g2 = gist.get_gist_features(frame[:,:,2])
        g = np.hstack((g0,g1,g2))
        pplot.plot(g)
        pplot.show()
        
        print ("processing time:",time.time() - ts)


# ----------------------------------------------------------------------------
