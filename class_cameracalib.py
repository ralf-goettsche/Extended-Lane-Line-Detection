import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from os import listdir
from os.path import isfile, join


class cameracalib:

    def __init__ (self, mode, deblev=0):
        self.mode = mode # 'single': Calibration on one dedicated picture
                         # 'multi':  Calibration on a set of pictures defined by dir location
        self.deblev = deblev  # '0': No debug information
                              # '1': Only textual debug information
                              # '2': All available debug information
                         
    def get_cameracalib (self, input, outfile=''):
        """
        Function for calibrating the camera based upon pictures under 'dirpath'
        Optionally, the results can be printed out_img
        
        Input: input -> For 'single' - Path plus file name
                        For 'multi'  - Path to pictues for calibration
               outfile -> Only 'single' - Path plus file name for undistorted picture
        Output: Camera matrix 'mtx' and camera distortion 'dst'
        """
        images = []
        imgpoints = []
        objpoints = []
        filenames = []
        # Definition of object points
        objp = np.zeros((9*6,3), np.float32)
        objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
        if self.mode == 'multi':
            # Get imagepoints of all suited pictures
            files = listdir(input)
            if self.deblev > 0:
                print("")
            for f in files:
                if isfile(join(input, f)):
                    file = join(input, f)
                    if self.deblev > 0:
                        print('{}: '.format(file), end='')
                    img = mpimg.imread(file)
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
                    # If chessboard corners could be generated in picture, store imagepoints,
                    # objectpoints and filename (for optional printout)
                    if ret == True:
                        img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                        images.append(img)
                        if self.deblev > 0:
                            print('integrated')
                        imgpoints.append(corners)
                        objpoints.append(objp)
                        filenames.append(file)
                    else:
                        if self.deblev > 0:
                            print('-')
        
        # For calibrating the camera on one specific picture (1st part)
        if self.mode == 'single':
            img = mpimg.imread(input)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            objpoints, imgpoints = [], []
            if ret == True:
                img = cv2.drawChessboardCorners(img, (9, 6), corners, ret)
                images.append(img)
                imgpoints.append(corners)
                objpoints.append(objp)
                filenames.append(input)
            else:
                print("ERROR: Chessboard couldn't be identified in picture! Abbortion!")
                exit()
        
        # Calculating camera matrix 'mtx', camera distortion 'dist', etc. (for both options(single or many pictures))
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        # For calibrating the camera on one specific picture, print out (2nd part)
        if self.mode == 'single':
            undistimg = cv2.undistort(img, mtx, dist, None, mtx)
            mpimg.imsave(fname=outfile, arr=undistimg)
            images.append(undistimg)
            filenames.append(outfile)

        # Optional output of all pictures and their undistorted one
        if self.deblev > 1:
            if self.mode == 'single':
                f, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10), sharey=True, num='Calib. Pics')
                axes[0].imshow(images[0])
                axes[0].set_title('Dist. Image\n{}'.format(filenames[0]), fontsize=10)
                axes[0].set_axis_off()
                axes[1].imshow(images[1])
                axes[1].set_title('Undist. Image\n{}'.format(filenames[1]), fontsize=10)
                axes[1].set_axis_off()
            if self.mode == 'multi':
                f, axes = plt.subplots(nrows=int(len(images)/2) + len(images)%2, ncols=4, figsize=(15, 10), sharey=True, num='Calib. Pics')
                for i in range(len(images)):
                    axes[int(i/2), i%2*2 + 0].imshow(images[i])
                    axes[int(i/2), i%2*2 + 0].set_title('Dist. Image\n{}'.format(filenames[i]), fontsize=10)
                    axes[int(i/2), i%2*2 + 0].set_axis_off()
                    axes[int(i/2), i%2*2 + 1].imshow(cv2.undistort(images[i], mtx, dist, None, mtx))
                    axes[int(i/2), i%2*2 + 1].set_title('Undist. Image\n{}'.format(filenames[i]), fontsize=10)
                    axes[int(i/2), i%2*2 + 1].set_axis_off()
                if (len(images)%2):
                    i += 1
                    axes[ int(i/2), i%2*2 + 0].set_axis_off()
                    axes[ int(i/2), i%2*2 + 1].set_axis_off()
                
            plt.show()
        
        # Handing back the camera matrix 'mtx' and the camera distortion 'dst'
        return (mtx, dist)
        
    def run(self, input, outfile=''):
        if self.deblev > 0:
            print("\nMode: \'{}\'".format(self.mode))
        
        mtx, dist = self.get_cameracalib(input=input, outfile=outfile)
        print('\nmtx =\n{}'.format(mtx))
        print('dist =\n{}\n'.format(dist))
    