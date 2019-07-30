import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage as sc

from moviepy.editor import VideoFileClip

from datetime import datetime


class lane_lines:

    def __init__ (self, mode='picture', output='deluxe', debuglevel=0):
        self.mode       = mode         # 'picture', 'video'
        self.output     = output       # 'none': Pure picture
                                       # 'info': Picture with information
                                       # 'small':  4 additional pics (undistorted pic, bird pic, filtered (all) bird pic, boxed or continous poly-fit)
                                       # 'medium':  7 additional pics (undistorted pic, bird pic, filtered (all) bird pic, hls(white)-filtered bird pic, yellow-filtered bird pic, boxed and continous poly-fitting)
                                       # 'deluxe': 12 additional pics (undistorted pic, bird pic, filtered (all) bird pic, hls(white)-filtered bird pic, yellow-filtered bird pic, boxed and continous poly-fitting, lab pics)
        self.deblev     = debuglevel   # 0: print only media with lines
                                       # 1: all above (aa) + run data
                                       # 2: (aa)
                                       # 3: (aa) + bird pic + grad pics
                                       # 4: (aa) + detailed pics (grad, poly)                                       
                                       # 5: plot all debug information
                                       #    (aa) + HSV pics + HSV bird pics

        # Params needed for movie generation
        self.first_pic_done     = False
        self.nosanitcnt         = 0
        self.sancntthres        = 4
        self.left_fit_glob      = []
        self.right_fit_glob     = []
        self.left_curverad      = 0.0
        self.left_curverad_old  = 0.0
        self.right_curverad     = 0.0
        self.right_curverad_old = 0.0
    
    def lcn (self, layer, xwindim, ywindim):
        """
        Local Contrast Normalization
        """
        minmat = np.zeros_like(layer, dtype=np.uint8)
        maxmat = np.zeros_like(layer, dtype=np.uint8)
        maxmat = sc.maximum_filter(layer, size=(ywindim,xwindim), mode='reflect')
        minmat = sc.minimum_filter(layer, size=(ywindim,xwindim), mode='reflect')

        extrdiff = ((maxmat - minmat) > 15) * (maxmat - minmat)
        nonexistsdiff = np.array(np.logical_not(extrdiff), dtype=np.uint8)
        existsdiff = np.array(np.logical_not(nonexistsdiff), dtype=np.uint8)
        
        locnormedlayer = np.zeros_like(layer, dtype=np.uint8)
        locnormedlayer = nonexistsdiff * layer + existsdiff * np.uint8(255 * np.divide((layer - minmat),extrdiff))            

        return locnormedlayer

    def norm (self, layer):
        """
        Selection of norm to be taken
        """
        normedlayer = self.lcn(layer,11,11)
        return normedlayer

    def hsvnorm (self, img, layertype = 'all'):
        """
        Norming the image by either saturation ('s') or value ('v') or both,
        the hue remains the same
        """
        normedimg = {
            'all':  lambda img: np.dstack((img[:,:,0],self.norm(np.array(img[:,:,1], dtype=np.uint8)),self.norm(np.array(img[:,:,2], dtype=np.uint8)))), 
            's':    lambda img: self.norm(np.array(img[:,:,1], dtype=np.uint8)),
            'v':    lambda img: self.norm(np.array(img[:,:,2], dtype=np.uint8))
        }[layertype](img)

        return normedimg
        
    def convert_image (self, img, method = 'rgb2gray'):
        """
        Function for color transformation of image 'img' according to 'method'
        
        Input: Image 'img', color transformation method 'method' 
        Output: The transformed image 'cvtimg'
        """
        cvtimg = {
            'rgb2hls':   lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2HLS),
            'rgb2hls,h':  lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,0],
            'rgb2hls,l':  lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,1],
            'rgb2hls,s':  lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2HLS)[:,:,2],
            'rgb2hsv':   lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2HSV),
            'rgb2hsv,h':  lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,0],
            'rgb2hsv,s':  lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,1],
            'rgb2hsv,v':  lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,2],
            'rgb2hsvnorm': lambda img: self.hsvnorm(cv2.cvtColor(img, cv2.COLOR_RGB2HSV),'all'),
            'rgb2hsvnorm,s': lambda img: self.hsvnorm(cv2.cvtColor(img, cv2.COLOR_RGB2HSV),'s'),
            'rgb2hsvnorm,v': lambda img: self.hsvnorm(cv2.cvtColor(img, cv2.COLOR_RGB2HSV),'v'),
            'rgb2bgr':   lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            'rgb2luv':   lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2LUV),
            'rgb2yuv':   lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2YUV),
            'rgb2ycrcb': lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb),
            'rgb2rgb':   lambda img: img,
            'rgb2gray':  lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
            'rgb2lab' :  lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2Lab),
            'rgb2lab,l' :  lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:,:,0],
            'rgb2lab,a' :  lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:,:,1],
            'rgb2lab,b' :  lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:,:,2],
            'bgr2hls':   lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2HLS),
            'bgr2hls,h':  lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,0],
            'bgr2hls,l':  lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,1],
            'bgr2hls,s':  lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,2],
            'bgr2hsv':   lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
            'bgr2hsv,h':  lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,0],
            'bgr2hsv,s':  lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,1],
            'bgr2hsv,v':  lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,2],
            'bgr2bgr':   lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2BGR),
            'bgr2luv':   lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2LUV),
            'bgr2yuv':   lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2YUV),
            'bgr2ycrcb': lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb),
            'bgr2bgr':   lambda img: img,
            'bgr2gray':  lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            'bgr2lab' :  lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2LAB),
            'bgr2lab,l' :  lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:,:,0],
            'bgr2lab,a' :  lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:,:,1],
            'bgr2lab,b' :  lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:,:,2]
        }[method](img)
        
        return cvtimg

    def calc_curvature(self, leftx, rightx, lefty, righty):
        """
        Function for calculating the curvature of a polynomial in meters
        (Taken from lecture with adoption of 'xm_per_pix')
        
        Input: X- and y-coordinates of the points building the left line ('leftx', 'lefty')
               and building the right line ('rightx', 'righty')
        Output: Curvature of the left line 'left_curverad' and right line 'right_curverad'
        """
        leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
        rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/600 # meters per pixel in x dimension

        y_eval_left = np.max(lefty)
        y_eval_right = np.max(righty)
        # Fit polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval_left*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval_right*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Curvature in meters
        return left_curverad, right_curverad
    
    def calc_xoffset(self, left_fit, right_fit, shape):
        """
        Function for calculating the offset of the car to the center of the road in meters
        
        Input: Parameter of the adopted polynom for the left line ('left_fit') and right line ('right_fit'),
               Dimension of the picture 'shape'
        Output: Offset 'xoffset' of the car to the center of the road in meters
        """
        xm_per_pix = 3.7/600 # meters per pixel in x dimension

        xlbottom = left_fit[0]*shape[0]**2 + left_fit[1]*shape[0] + left_fit[2]
        xrbottom = right_fit[0]*shape[0]**2 + right_fit[1]*shape[0] + right_fit[2]
        
        xoffset = ((xlbottom + xrbottom) - shape[1])/2 * xm_per_pix
        return xoffset

    def plotimg (self, img, method='rgb2rgb'):
        """
        Plotting one image with according color map
        
        Packages: matplotlib.pyplot, cv2, re
        Input: Image 'img', color map 'method'
        """
        plt.figure('plotimg')
        if (len(img.shape) == 3):
            if (re.match(".*2rgb.*", method)):
                plt.imshow(img)
            elif (re.match(".*2hsv.*", method)):
                plt.imshow(img, cmap='hsv')
            elif (re.match(".*2bgr.*", method)):
                plt.imshow(img, cmap='brg')
            else:
                plt.imshow(img)
        else:
            plt.imshow(img, cmap='gray')
        
        plt.show()

    def plot2img (self, img, cvtimg, method='rgb2rgb'):
        """
        Plotting two images next to each other (mainly for comparison reason) in according color map
        
        Input: Images 'img' and 'cvtimg', color map 'method'
        """
        f, (axe1, axe2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 10), num='plot2img')
        axe1.imshow(img)
        axe1.set_title('Orig', fontsize=15)
        if (len(cvtimg.shape) == 3):
            if (re.match(".*2rgb.*", method)):
                axe2.imshow(cvtimg)
            elif (re.match(".*2hsv.*", method)):
                axe2.imshow(cvtimg, cmap='hsv')
            elif (re.match(".*2bgr.*", method)):
                axe2.imshow(cvtimg, cmap='brg')
            else:
                axe2.imshow(cvtimg)
        else:
            axe2.imshow(cvtimg, cmap='gray')

        axe2.set_title('Converted Image', fontsize=15)
        plt.show()

    def plot_lanelines(self, pltimg, left_fit, right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy):
        """
        Plotting one image with adopted polynomes for left and rigth line and highlighted lane lines
        
        Input: Warped image 'pltimg', parameters for the adopted polynomes for left and right line ('left_fit', 'right_fit'),
               indices of line points ('left_lane_inds', 'right_lane_inds') in nonzero arrays ('nonzerox', 'nonzeroy')
        """
        # Calculating the base in y-parameters
        ploty = np.linspace(0, pltimg.shape[0]-1, pltimg.shape[0] )
        # Calculating the points on the polynoms for left and right line in x-parameters
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        leftpoly = np.array([left_fitx, ploty], np.int32).T
        leftpoly = leftpoly.reshape((-1,1,2))
        rightpoly = np.array([right_fitx, ploty], np.int32).T
        rightpoly = rightpoly.reshape((-1,1,2))

        # Coloring the points of the lines
        pltimg[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        pltimg[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        cv2.polylines(pltimg, [leftpoly], False, (255,255,0), 6)
        cv2.polylines(pltimg, [rightpoly], False, (255,255,0), 6)
        if self.deblev > 3:
            plt.figure('Fitted Lane Lines')
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.imshow(pltimg)
            plt.xlim(0, pltimg.shape[1])
            plt.ylim(pltimg.shape[0], 0)
            plt.show()
        
        return pltimg

    def plot_gallery_small(self, markimage, undistimage, birdimage, out_img, boxedpolyimg, contpolyimg, \
                           left_curverad, right_curverad, xoffset, sanity, vp):

        debug_image = np.zeros_like(cv2.resize(markimage,(1920,1080)))
        
        # The result picture with lane lines
        debug_image[360:1080,0:1280,:] = markimage

        # Add curvature, offset, and sanity information into final image
        str =("left curve:   {:7.2f}m".format(left_curverad))
        cv2.putText(debug_image, str, (50,80), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.2, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
        str =("right curve: {:7.2f}m".format(right_curverad))
        cv2.putText(debug_image, str, (50,120), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.2, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
        str =("Center offset:  {:5.2f}m".format(xoffset))
        cv2.putText(debug_image, str, (50,160), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.2, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
        str =("Sanity: {}".format(sanity))
        cv2.putText(debug_image, str, (50,200), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.2, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
        str =("VP: ({},{})".format(vp[0][0],vp[1][0]))
        cv2.putText(debug_image, str, (260,200), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.2, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
        cv2.rectangle(debug_image, (2,2), (640, 360), color=(255,255,255), thickness=4)

        # undistorted image
        sx_undistimage = cv2.resize(undistimage, (640, 360))
        debug_image[0:360,640:1280,:] = sx_undistimage
        cv2.putText(debug_image, "undistorted".format(0.5), (860,345), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.0, color=(255,150,0), thickness=1, lineType = cv2.LINE_AA)
        cv2.rectangle(debug_image, (640,2), (1280, 360), color=(255,255,255), thickness=4)
        
        # bird's view image
        sx_birdimage = cv2.resize(birdimage, (640, 360))
        debug_image[0:360,1280:1920,:] = sx_birdimage
        cv2.putText(debug_image, "bird's view".format(0.5), (1520,345), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.0, color=(255,150,0), thickness=1, lineType = cv2.LINE_AA)
        cv2.rectangle(debug_image, (1280,2), (1918, 360), color=(255,255,255), thickness=4)
        
        # complete filtered image
        sx_filtered_image = cv2.resize(out_img, (640, 360))
        debug_image[360:720,1280:1920,:] = sx_filtered_image
        cv2.putText(debug_image, "bird's view, filtered".format(0.5), (1480,705), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(255,150,0), thickness=1, lineType = cv2.LINE_AA)
        cv2.rectangle(debug_image, (1280,360), (1918, 720), color=(255,255,255), thickness=4)
        
        if 1:
            # boxed poly-fitting image
            sx_poly_image = cv2.resize(boxedpolyimg, (640, 360))
            debug_image[720:1080,1280:1920,:] = sx_poly_image
            cv2.putText(debug_image, "boxed poly-fit".format(0.5), (1510,1065), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(255,150,0), thickness=1, lineType = cv2.LINE_AA)
            cv2.rectangle(debug_image, (1280,720), (1918, 1078), color=(255,255,255), thickness=4)
        else:
            # continous poly-fitting image
            sx_poly_image = cv2.resize(contpolyimg, (640, 360))
            debug_image[720:1080,1280:1920,:] = sx_poly_image
            cv2.putText(debug_image, "continous poly-fit".format(0.5), (1480,1065), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(255,150,0), thickness=1, lineType = cv2.LINE_AA)
            cv2.rectangle(debug_image, (1280,720), (1918, 1078), color=(255,255,255), thickness=4)
    
        return debug_image

    def plot_gallery_medium(self, markimage, undistimage, birdimage, out_img, boxedpolyimg, contpolyimg, \
                            combined_hls_filter_img, yellow_filter_img, \
                            left_curverad, right_curverad, xoffset, sanity, vp):
    
        debug_image = np.zeros_like(cv2.resize(markimage,(1600,900)))
        
        # The result picture with lane lines
        debug_image[180:900,0:1280,:] = markimage

        # Add curvature, offset, and sanity information into final image
        str =("left curve:   {:7.2f}m".format(left_curverad))
        cv2.putText(debug_image, str, (50,40), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
        str =("right curve: {:7.2f}m".format(right_curverad))
        cv2.putText(debug_image, str, (50,80), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
        str =("Center offset:  {:5.2f}m".format(xoffset))
        cv2.putText(debug_image, str, (50,120), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
        str =("Sanity: {}".format(sanity))
        cv2.putText(debug_image, str, (50,160), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
        str =("VP: ({},{})".format(vp[0][0],vp[1][0]))
        cv2.putText(debug_image, str, (200,160), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
        cv2.rectangle(debug_image, (2,2), (640, 180), color=(255,255,255), thickness=4)

        # undistorted image
        sx_undistimage = cv2.resize(undistimage, (320, 180))
        debug_image[0:180,640:960,:] = sx_undistimage
        cv2.putText(debug_image, "undistorted".format(0.5), (755,170), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255,150,0), thickness=1, lineType = cv2.LINE_AA)
        cv2.rectangle(debug_image, (640,2), (960, 180), color=(255,255,255), thickness=4)
        
        # bird's view image
        sx_birdimage = cv2.resize(birdimage, (320, 180))
        debug_image[0:180,960:1280,:] = sx_birdimage
        cv2.putText(debug_image, "bird's view".format(0.5), (1080,170), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255,150,0), thickness=1, lineType = cv2.LINE_AA)
        cv2.rectangle(debug_image, (960,2), (1280, 180), color=(255,255,255), thickness=4)
        
        # complete filtered image
        sx_filtered_image = cv2.resize(out_img, (320, 180))
        debug_image[0:180,1280:1600,:] = sx_filtered_image
        cv2.putText(debug_image, "bird's view, filtered".format(0.5), (1375,170), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.4, color=(255,150,0), thickness=1, lineType = cv2.LINE_AA)
        cv2.rectangle(debug_image, (1280,2), (1598, 180), color=(255,255,255), thickness=4)
        
        # combined-hls-filtered image
        sx_filtered_combined_hls_image = cv2.resize(combined_hls_filter_img, (320, 180))
        debug_image[180:360,1280:1600,:] = sx_filtered_combined_hls_image
        cv2.putText(debug_image, "bird's view, filtered comb_hls".format(0.5), (1380,350), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.3, color=(255,150,0), thickness=1, lineType = cv2.LINE_AA)
        cv2.rectangle(debug_image, (1280,180), (1598, 360), color=(255,255,255), thickness=4)
        
        # yellow-filtered image
        sx_filtered_yellow_image = cv2.resize(yellow_filter_img, (320, 180))
        debug_image[360:540,1280:1600,:] = sx_filtered_yellow_image
        cv2.putText(debug_image, "bird's view, filtered yellow".format(0.5), (1380,530), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.3, color=(255,150,0), thickness=1, lineType = cv2.LINE_AA)
        cv2.rectangle(debug_image, (1280,360), (1598, 540), color=(255,255,255), thickness=4)
        
        # boxed poly-fitting image
        sx_poly_box_image = cv2.resize(boxedpolyimg, (320, 180))
        debug_image[540:720,1280:1600,:] = sx_poly_box_image
        cv2.putText(debug_image, "boxed poly-fit".format(0.5), (1395,710), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.4, color=(255,150,0), thickness=1, lineType = cv2.LINE_AA)
        cv2.rectangle(debug_image, (1280,540), (1598, 720), color=(255,255,255), thickness=4)
        
        # continous poly-fitting image
        sx_poly_cont_image = cv2.resize(contpolyimg, (320, 180))
        debug_image[720:900,1280:1600,:] = sx_poly_cont_image
        cv2.putText(debug_image, "continous poly-fit".format(0.5), (1380,890), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.4, color=(255,150,0), thickness=1, lineType = cv2.LINE_AA)
        cv2.rectangle(debug_image, (1280,720), (1598, 898), color=(255,255,255), thickness=4)
    
        return debug_image
    
    def plot_galleray_deluxe(self, markimage, undistimage, birdimage, out_img, boxedpolyimg, contpolyimg, \
                             combined_hls_filter_img, yellow_filter_img, \
                             lab_l_filter_img, lab_b_filter_img, gradx_hls_l_filter_img, gradx_hls_s_filter_img, white_filter_img, \
                             left_curverad, right_curverad, xoffset, sanity, vp):

        debug_image = np.zeros_like(cv2.resize(markimage,(1920,900)))
        
        # The result picture with lane lines
        debug_image[180:900,0:1280,:] = markimage

        # Add curvature, offset, and sanity information into final image
        str =("left curve:   {:7.2f}m".format(left_curverad))
        cv2.putText(debug_image, str, (50,40), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
        str =("right curve: {:7.2f}m".format(right_curverad))
        cv2.putText(debug_image, str, (50,80), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
        str =("Center offset:  {:5.2f}m".format(xoffset))
        cv2.putText(debug_image, str, (50,120), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
        str =("Sanity: {}".format(sanity))
        cv2.putText(debug_image, str, (50,160), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
        str =("VP: ({},{})".format(vp[0][0],vp[1][0]))
        cv2.putText(debug_image, str, (200,160), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.8, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
        cv2.rectangle(debug_image, (2,2), (640, 180), color=(255,255,255), thickness=4)

        # undistorted image
        sx_undistimage = cv2.resize(undistimage, (320, 180))
        debug_image[0:180,640:960,:] = sx_undistimage
        cv2.putText(debug_image, "undistorted".format(0.5), (755,170), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255,150,0), thickness=1, lineType = cv2.LINE_AA)
        cv2.rectangle(debug_image, (640,2), (960, 180), color=(255,255,255), thickness=4)
        
        # bird's view image
        sx_birdimage = cv2.resize(birdimage, (320, 180))
        debug_image[0:180,960:1280,:] = sx_birdimage
        cv2.putText(debug_image, "bird's view".format(0.5), (1080,170), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(255,150,0), thickness=1, lineType = cv2.LINE_AA)
        cv2.rectangle(debug_image, (960,2), (1280, 180), color=(255,255,255), thickness=4)
        
        # combined-hls-filtered image
        sx_filtered_combined_hls_image = cv2.resize(combined_hls_filter_img, (320, 180))
        debug_image[180:360,1280:1600,:] = sx_filtered_combined_hls_image
        cv2.putText(debug_image, "bird's view, filtered hls".format(0.5), (1380,350), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.3, color=(255,150,0), thickness=1, lineType = cv2.LINE_AA)
        cv2.rectangle(debug_image, (1280,180), (1598, 360), color=(255,128,0), thickness=4)
        
        # lab-l-filtered image
        sx_filtered_lab_l_image = cv2.resize(lab_l_filter_img, (320, 180))
        debug_image[540:720,1600:1920,:] = sx_filtered_lab_l_image
        cv2.putText(debug_image, "bird's view, filtered lab_l".format(0.5), (1680,710), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.3, color=(255,150,0), thickness=1, lineType = cv2.LINE_AA)
        cv2.rectangle(debug_image, (1600,540), (1918, 720), color=(0,0,255), thickness=4)
        
        # lab-b-filtered image
        sx_filtered_lab_b_image = cv2.resize(lab_b_filter_img, (320, 180))
        debug_image[720:900,1600:1920,:] = sx_filtered_lab_b_image
        cv2.putText(debug_image, "bird's view, filtered lab_b".format(0.5), (1680,890), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.3, color=(255,150,0), thickness=1, lineType = cv2.LINE_AA)
        cv2.rectangle(debug_image, (1600,720), (1918, 898), color=(0,0,255), thickness=4)

        # boxed poly-fitting image
        sx_poly_box_image = cv2.resize(boxedpolyimg, (320, 180))
        debug_image[540:720,1280:1600,:] = sx_poly_box_image
        cv2.putText(debug_image, "boxed poly-fit".format(0.5), (1395,710), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.4, color=(255,150,0), thickness=1, lineType = cv2.LINE_AA)
        cv2.rectangle(debug_image, (1280,540), (1598, 720), color=(0,255,0), thickness=4)
        
        # continous poly-fitting image
        sx_poly_cont_image = cv2.resize(contpolyimg, (320, 180))
        debug_image[720:900,1280:1600,:] = sx_poly_cont_image
        cv2.putText(debug_image, "continous poly-fit".format(0.5), (1380,890), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.4, color=(255,150,0), thickness=1, lineType = cv2.LINE_AA)
        cv2.rectangle(debug_image, (1280,720), (1598, 898), color=(0,255,0), thickness=4)

        # gradx-hls-l-filtered image
        sx_filtered_gradx_hls_l_image = cv2.resize(gradx_hls_l_filter_img, (320, 180))
        debug_image[0:180,1600:1920,:] = sx_filtered_gradx_hls_l_image
        cv2.putText(debug_image, "bird's view, filtered gradx_hls_l".format(0.5), (1700,170), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.3, color=(255,150,0), thickness=1, lineType = cv2.LINE_AA)
        cv2.rectangle(debug_image, (1600,2), (1918, 180), color=(128,128,128), thickness=4)
        
        # gradx-hls-s-filtered image
        sx_filtered_gradx_hls_s_image = cv2.resize(gradx_hls_s_filter_img, (320, 180))
        debug_image[180:360,1600:1920,:] = sx_filtered_gradx_hls_s_image
        cv2.putText(debug_image, "bird's view, filtered gradx_hls_s".format(0.5), (1700,350), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.3, color=(255,150,0), thickness=1, lineType = cv2.LINE_AA)
        cv2.rectangle(debug_image, (1600,180), (1918, 360), color=(128,128,128), thickness=4)

        # white-filtered image
        sx_filtered_white_image = cv2.resize(white_filter_img, (320, 180))
        debug_image[360:540,1600:1920,:] = sx_filtered_white_image
        cv2.putText(debug_image, "bird's view, filtered white".format(0.5), (1700,530), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.3, color=(255,150,0), thickness=1, lineType = cv2.LINE_AA)
        cv2.rectangle(debug_image, (1600,360), (1918, 540), color=(255,255,0), thickness=4)        
        
        # yellow-filtered image
        sx_filtered_yellow_image = cv2.resize(yellow_filter_img, (320, 180))
        debug_image[360:540,1280:1600,:] = sx_filtered_yellow_image
        cv2.putText(debug_image, "bird's view, filtered yellow".format(0.5), (1380,530), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.3, color=(255,150,0), thickness=1, lineType = cv2.LINE_AA)
        cv2.rectangle(debug_image, (1280,360), (1598, 540), color=(255,255,0), thickness=4)
        
        # complete filtered image
        sx_filtered_image = cv2.resize(out_img, (320, 180))
        debug_image[0:180,1280:1600,:] = sx_filtered_image
        cv2.putText(debug_image, "bird's view, filtered".format(0.5), (1375,170), cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.4, color=(255,150,0), thickness=1, lineType = cv2.LINE_AA)
        cv2.rectangle(debug_image, (1280,2), (1598, 180), color=(255,0,0), thickness=4)
    
        return debug_image
        
    def gen_img_w_marks(self, img, bin_warp, left_fit, right_fit, Minv):
        """
        Generating image with marked lane area
        (Mostly taken from lecture)
        
        Input: Image 'img', binary filtered and warped image 'bin_warp',
               parameters of the polynoms for left and right line ('left_fit', 'right_fit')
               and the unwarp matrix 'Minv'
        Output: Image with marked lane area
        """
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(bin_warp).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Calculating the base in y-parameters
        ploty = np.linspace(0, bin_warp.shape[0]-1, bin_warp.shape[0] )
        # Calculating the points on the polynoms for left and right line in x-parameters
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the marked lane area onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        marks = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
        # Combine the result with the original image
        img = cv2.addWeighted(img, 1, marks, 0.3, 0)
        
        return img

    def color_mask (self, img, ll=[0,0,0], ul=[0,0,0], method='rgb2rgb'):
        conv_img = self.convert_image(img, method=method)
        mask = np.zeros_like(conv_img[:,:,1], dtype=np.uint8)
        mask[(conv_img[:,:,0] >= ll[0]) & (conv_img[:,:,0] <= ul[0]) &
             (conv_img[:,:,1] >= ll[1]) & (conv_img[:,:,1] <= ul[1]) &
             (conv_img[:,:,2] >= ll[2]) & (conv_img[:,:,2] <= ul[2])] = 1
        return mask
    
    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0,255), method='gray'):
        """
        Edge detection in x or y direction with according color transformation in certain threshold region
        
        Input: RGB image 'img', the detection direction 'orient', filter size 'sobel_kernel',
               threshold range 'thresh', color transformation 'method'
        Output: Binary filtered image 'binary_output'
        """
        # Color transformation
        gray = self.convert_image(img, method=method)
        # Edge detection according to orientation with chosen filter size 
        if orient == 'x':
            devimg = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        elif orient== 'y':
            devimg = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        
        # Transformation into image data
        devimg = np.absolute(devimg)
        devimg = np.uint8(255*devimg/np.max(devimg))
        
        # Filtering based upon threshold range
        binary_output = np.zeros_like(devimg, dtype=np.uint8)
        binary_output[(devimg >= thresh[0]) & (devimg <= thresh[1])] = 1
        
        return binary_output

    def mag_thresh(self, img, sobel_kernel=3, mag_thresh=(0, 255), method='gray'):
        """
        Magnitude calculation of x- and y-edges with according color transformation in certain threshold region
        
        Input: RGB image 'img', filter size 'sobel_kernel', threshold range 'mag_thresh', color transformation 'method'
        Output: Magnitude binary filtered image 'binary_output'
        """
        # Color transformation
        gray = self.convert_image(img, method=method)
        # Edge detection in x- and y-direction
        devximg = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        devyimg = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        
        # Magnitude of edge detection calculation and transformation into image data
        absimg = np.sqrt(devximg**2 + devyimg**2 )
        absimg = np.uint8(255*absimg/np.max(absimg))
        
        # Filtering based upon threshold range
        binary_output = np.zeros_like(absimg, dtype=np.uint8)
        binary_output[(absimg >= mag_thresh[0]) & (absimg <= mag_thresh[1])] = 1
        
        return binary_output

    def dir_threshold(self, img, sobel_kernel=3, dir_thresh=(0, np.pi/2), method='gray'):
        """
        Edge direction calculation with according color transformation in certain threshold region
        
        Input: RGB image 'img', filter size 'sobel_kernel', threshold range 'dir_thresh', color transformation 'method'
        Output: Edge direction filtered image 'binary_output'
        """
        # Color transformation
        gray = self.convert_image(img, method=method)
        # Edge detection in x- and y-direction
        devximg = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        devyimg = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        
        # Calculation of direction of the edges
        absximg = np.absolute(devximg)
        absyimg = np.absolute(devyimg)
        dirdevimg = np.arctan2(absyimg, absximg)
        
        # Filtering based upon direction threshold range
        binary_output = np.zeros_like(dirdevimg, dtype=np.uint8)
        binary_output[(dirdevimg >= dir_thresh[0]) & (dirdevimg <= dir_thresh[1])] = 1
        
        return binary_output
        
    def grad_color_filter (self, image, imagenorm):
        """
        Filter and color transformation showing most promising results
        
        Input: RGB Image 'image'
        Output: Binary filtered image 'combined'
        """
        ### First filter definition based upon the l-layer of hls transformation
        method = 'rgb2hls,l'
        # Filtering l-image with similar parameters of lecture
        gradx_hls_l = self.abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(20, 200), method=method)
        grady_hls_l = self.abs_sobel_thresh(image, orient='y', sobel_kernel=3, thresh=(20, 100), method=method)
        mag_binary_hls_l = self.mag_thresh(image, sobel_kernel=3, mag_thresh=(30,100), method=method)
        dir_binary_hls_l = self.dir_threshold(image, sobel_kernel=15, dir_thresh=(0.7, 1.3), method=method)
        ## For bird's view
        combined_hls_l  = np.zeros_like(gradx_hls_l)
        combined_hls_l[(gradx_hls_l == 1)] = 1
        
        ### Second filter definition based upon the s-layer of hls transformation
        method = 'rgb2hls,s'
        # Filtering s-image, parameter of gradients have been lowered for better results 
        gradx_hls_s = self.abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(20, 200), method=method)
        grady_hls_s = self.abs_sobel_thresh(image, orient='y', sobel_kernel=3, thresh=(20, 100), method=method)
        mag_binary_hls_s = self.mag_thresh(image, sobel_kernel=3, mag_thresh=(30,100), method=method)
        dir_binary_hls_s = self.dir_threshold(image, sobel_kernel=15, dir_thresh=(0.7, 1.3), method=method)
        ## For bird's view
        combined_hls_s  = np.zeros_like(gradx_hls_s)
        combined_hls_s[(gradx_hls_s == 1)] = 1
       
        
        ### color-filtering
        # white lane lines
        ###
        white_hsv_low  = np.array([  0, 200,   0])
        white_hsv_high = np.array([255, 255, 255])
        white_filter = self.color_mask(image, ll=white_hsv_low, ul=white_hsv_high, method='rgb2hls')
        ###
        # yellow lane lines
        ###
        yellow_hsv_low  = np.array([ 10, 176, 150])
        yellow_hsv_high = np.array([ 40, 255, 255])
        yellow_filter = self.color_mask(imagenorm, ll=yellow_hsv_low, ul=yellow_hsv_high, method='rgb2hsv')

        ### LAB
        filter_lab_l = self.abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(10, 250), method='rgb2lab,l')
        filter_lab_b = self.abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(20, 250), method='rgb2lab,b')
        
        
        ### Combination
        # Combining color filters
        color_filter = np.zeros_like(white_filter)
        color_filter[(white_filter == 1) | (yellow_filter == 1)] = 1
        
        # Combining hls filters
        combined_hls  = np.zeros_like(combined_hls_s)
        combined_hls[(combined_hls_l == 1) | (combined_hls_s == 1) | (filter_lab_l == 1) | (filter_lab_b == 1)] = 1
 
        # Combining all filters
        combined  = np.zeros_like(combined_hls_l)
        combined[((combined_hls_l == 1) | (combined_hls_s == 1)) | (color_filter == 1)] = 1
       
        if self.deblev > 2:
            f, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10), (ax11, ax12), (ax13, ax14)) = plt.subplots(7, 2, figsize=(24, 9), num='Grad Pics')
            f.tight_layout()
            ax1.imshow(combined_hls, cmap='gray')
            ax1.set_title('Combined Image, hls', fontsize=20)
            ax2.imshow(combined, cmap='gray')
            ax2.set_title('Combined Image', fontsize=20)
            ax3.imshow(gradx_hls_l, cmap='gray')
            ax3.set_title('Gradx, hls_l', fontsize=20)
            ax4.imshow(grady_hls_l, cmap='gray')
            ax4.set_title('Grady, hls_l', fontsize=20)
            ax5.imshow(gradx_hls_s, cmap='gray')
            ax5.set_title('Gradx, hls_s', fontsize=20)
            ax6.imshow(grady_hls_s, cmap='gray')
            ax6.set_title('Grady, hls_s', fontsize=20)
            ax7.imshow(mag_binary_hls_l, cmap='gray')
            ax7.set_title('Magnitude Gradient Image, hls_l', fontsize=20)
            ax8.imshow(dir_binary_hls_l, cmap='gray')
            ax8.set_title('Directed Gradient Image, hls_l', fontsize=20)
            ax9.imshow(mag_binary_hls_s, cmap='gray')
            ax9.set_title('Magnitude Gradient Image, hls_s', fontsize=20)
            ax10.imshow(dir_binary_hls_s, cmap='gray')
            ax10.set_title('Directed Gradient Image, hls_s', fontsize=20)
            ax11.imshow(yellow_filter, cmap='gray')
            ax11.set_title('Yellow Color Filter, hsv', fontsize=20)
            ax12.imshow(white_filter, cmap='gray')
            ax12.set_title('White Color Filter, hsv', fontsize=20)
            ax13.imshow(filter_lab_l, cmap='gray')
            ax13.set_title('Filter LAB, L', fontsize=20)
            ax14.imshow(filter_lab_b, cmap='gray')
            ax14.set_title('Filter LAB, B', fontsize=20)

            plt.show()
        
        return combined, combined_hls, white_filter, yellow_filter, gradx_hls_l, gradx_hls_s, filter_lab_l, filter_lab_b

    def corners_unwarp(self, img):
        """
        Warping image by just considering region of interest
        
        Input: Image 'img'
        Output: Warped image 'warped', transformation matrix 'M' and its inverse 'Minv'
        """
        imgx = img.shape[1]
        imgy = img.shape[0]
        
        ## Definition of rectangle inside of image to be transformed
        src = np.float32([[-104.67, 720.00], [572.00, 450.00], [717.00, 450.00], [1474.50, 720.00]]) # vp calc, straight_line2, dx=30
        vp = np.array([[640],[421]])
        
        ## Definition of target rectangle (full image area)
        dst = np.float32([[0,720], [0,0], [1280,0], [1280,720]])
        
        # Calculation of tranformation matrices
        M    = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        # Image transformation
        warped = cv2.warpPerspective(img, M, (imgx, imgy), flags=cv2.INTER_LINEAR)
        
        return warped, M, Minv, vp

    def lineidx_detection_per_window (self, bin_warped_img, nonzerox, nonzeroy, nwindows=13):
        """
        Line detection based upon window methodology
        (Taken from lectures and modified with regards to window imaging)
        
        Input: Binary filtered and warped image 'bin_warped_img', x-indices (nonzerox) and y-indices (nonzeroy)
               of non-zero points in binary filtered and warped image, number of windows 'nwindows' used for line detection
        Output: Location of points of the left line ('left_lane_inds') and right line ('right_lane_inds'),
                image with the windows 'win_img' for later optional print out
        """
        # Determining the histogram and hence the area of the lines in the image incl. optional print out
        histogram = np.sum(bin_warped_img[bin_warped_img.shape[0]//2:,:], axis=0)
        if self.deblev > 3:
            plt.figure('Lane Line Histogram')
            plt.plot(histogram)
            plt.show()
        
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = np.int(bin_warped_img.shape[0]/nwindows)
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100 # *50, *100*, 70
        # Set minimum number of pixels found to recenter window
        minpix = 150 #50, *100*, *150, 200
        # Create empty lists to receive left and right line pixel indices
        left_lane_inds = []
        right_lane_inds = []
        # Image for inserting the windows later on
        win_img = np.zeros_like(np.uint8(np.dstack((bin_warped_img,bin_warped_img,bin_warped_img))))

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = bin_warped_img.shape[0] - (window+1)*window_height
            win_y_high = bin_warped_img.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(win_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            [0,255,0], 2) 
            cv2.rectangle(win_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            [0,255,0], 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        return left_lane_inds, right_lane_inds, win_img

    def lineidx_detection_per_fit(self, bin_warped_img, nonzerox, nonzeroy, left_fit, right_fit, margin=100): # *75, *100*, 50
        """
        Line detection based upon fitting methodolgy (using existing fitting and adopting it)
        (Taken from lectures and modified with regards to search area imaging)
        
        Input: Binary filtered and warped image 'bin_warped_img', x-indices (nonzerox) and y-indices (nonzeroy)
               of non-zero points in binary filtered and warped image, parameters of the polynoms
               representing the left ('left_fit') and right line ('right_fit'), 'margin' of the search window
        Output: Location of points of the left line ('left_lane_inds') and right line ('right_lane_inds'),
                image with the search areas 'win_img' for later optional print out
        """
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
        left_fit[1]*nonzeroy + left_fit[2] + margin))) 

        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
        right_fit[1]*nonzeroy + right_fit[2] + margin)))  
        
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        ploty = np.linspace(0, bin_warped_img.shape[0]-1, bin_warped_img.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                      ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                      ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        win_img = np.uint8(np.dstack((bin_warped_img, bin_warped_img, bin_warped_img))*255)
        cv2.fillPoly(win_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(win_img, np.int_([right_line_pts]), (0,255, 0))

        return left_lane_inds, right_lane_inds, win_img

    def sanity_check(self, left_fit, right_fit, left_fit_glob, right_fit_glob, left_curverad, right_curverad, left_curverad_old, right_curverad_old, shape):
        """
        Function for the sanity check of determined lane detection
         
        Input: Parameter of the polynoms representing the left ('left_fit') and right line ('right_fit'),
               the calculated curvature of left ('left_curverad') and right line ('right_curverad'),
               image dimensions 'shape'
        Output: Boolean value 'sanity'
        """
        curvethres = 0.02

        if ((left_curverad > curvethres) & (right_curverad > curvethres)):
            curvethresok = 1
        else:
            curvethresok = 0
        
        # We analyze the difference at the x-coordinate...
        xdelta = 15
        # but the polynome is a function of y
        y = shape[0]
        x_left_fit = left_fit[0]*(y**2) + left_fit[1]*y + left_fit[2]
        x_right_fit = right_fit[0]*(y**2) + right_fit[1]*y + right_fit[2]
        x_left_fit_glob = left_fit_glob[0]*(y**2) + left_fit_glob[1]*y + left_fit_glob[2]
        x_right_fit_glob = right_fit_glob[0]*(y**2) + right_fit_glob[1]*y + right_fit_glob[2]
        if ((abs(x_left_fit - x_left_fit_glob) <= xdelta) & (abs(x_right_fit - x_right_fit_glob) <= xdelta)):
            xdiffok = 1
        else:
            xdiffok = 0
            
        curvedifffac = 100
        if ((left_curverad_old/curvedifffac < left_curverad) & (left_curverad < left_curverad_old*curvedifffac) &
            (right_curverad_old/curvedifffac < right_curverad) & (right_curverad < right_curverad_old*curvedifffac)):
            curvediffok = 1
        else:
            curvediffok = 0
             
        # Determining overall sanity
        sanity = curvethresok & xdiffok & curvediffok
        
        return sanity
        
    def process_single_image(self, media):
 
        # Values for camera matrix and camera distortion
        mtx = np.array([[  1.15396093e+03,   0.00000000e+00,   6.69705359e+02],
               [  0.00000000e+00,   1.14802495e+03,   3.85656232e+02],
               [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
        dist = np.array([[ -2.41017968e-01,  -5.30720497e-02,  -1.15810318e-03,  -1.28318543e-04, 2.67124302e-02]])
        
        image = mpimg.imread(media)
        
        # Undistorting image
        undistimage = cv2.undistort(image, mtx, dist, None, mtx)
        #plotimg(undistimage)

        hsvimgnorm  = self.convert_image(undistimage, method='rgb2hsvnorm')
        rgbimgnorm  = cv2.cvtColor(hsvimgnorm, cv2.COLOR_HSV2RGB)
        
        if self.deblev > 4:
            hsvimg = self.convert_image(undistimage, method='rgb2hsv')
            hsvimgh = hsvimg[:,:,0]
            hsvimgs = hsvimg[:,:,1]
            hsvimgv = hsvimg[:,:,2]

            hsvimghnorm = hsvimgnorm[:,:,0]
            hsvimgsnorm = hsvimgnorm[:,:,1]
            hsvimgvnorm = hsvimgnorm[:,:,2]

            f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(24, 9), num='HSV Pics.')
            f.tight_layout()
            ax1.imshow(hsvimg, cmap='hsv')
            ax1.set_title('HSV img', fontsize=20)
            ax2.imshow(hsvimgh, cmap='hsv')
            ax2.set_title('HSV-H map', fontsize=20)
            ax3.imshow(hsvimgs, cmap='gray')
            ax3.set_title('HSV-S map', fontsize=20)
            ax4.imshow(hsvimgv, cmap='gray')
            ax4.set_title('HSV-V map', fontsize=20)
            ax5.imshow(hsvimg, cmap=plt.get_cmap('hsv'))
            ax5.set_title('HSV-Norm', fontsize=20)
            ax6.imshow(hsvimghnorm, cmap='hsv')
            ax6.set_title('HSV-H-Norm', fontsize=20)
            ax7.imshow(hsvimgsnorm, cmap='gray')
            ax7.set_title('HSV-S-Norm', fontsize=20)
            ax8.imshow(hsvimgvnorm, cmap='gray')
            ax8.set_title('HSV-V-Norm', fontsize=20)

            plt.show()
        
        # Generate the warped image
        birdimagenorm, perspective_M, Minv, vp  = self.corners_unwarp(rgbimgnorm)
        birdimage, perspective_M, Minv, vp  = self.corners_unwarp(undistimage)
        
        if self.deblev > 2:
            self.plot2img(image, birdimage)

        if self.deblev > 4:
            hsvimg = self.convert_image(birdimage,method='rgb2hsv')
            hsvimgh = hsvimg[:,:,0]
            hsvimgs = hsvimg[:,:,1]
            hsvimgv = hsvimg[:,:,2]

            hsvimgnorm  = self.convert_image(birdimagenorm,method='rgb2hsv')
            hsvimghnorm = hsvimgnorm[:,:,0]
            hsvimgsnorm = hsvimgnorm[:,:,1]
            hsvimgvnorm = hsvimgnorm[:,:,2]

            f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12)) = plt.subplots(3, 4, figsize=(24, 9), num='HSV bird Pics')
            f.tight_layout()
            #ax1.imshow(image)
            #ax1.set_title('Original Image', fontsize=20)
            ax1.imshow(hsvimg, cmap='hsv')
            ax1.set_title('HSV img', fontsize=20)
            ax2.imshow(hsvimgh, cmap='hsv')
            ax2.set_title('HSV-H map', fontsize=20)
            #ax3.imshow(hsvimgs, cmap='gray')
            ax3.imshow(hsvimgs, cmap='Set3')
            ax3.set_title('HSV-S map', fontsize=20)
            #ax4.imshow(hsvimgv, cmap='gray')
            ax4.imshow(hsvimgv, cmap='Set3')
            ax4.set_title('HSV-V map', fontsize=20)
            ax5.imshow(hsvimgnorm, cmap='hsv')
            ax5.set_title('HSV-Norm', fontsize=20)
            ax6.imshow(hsvimghnorm, cmap='hsv')
            ax6.set_title('HSV-H-Norm', fontsize=20)
            #ax7.imshow(hsvimgsnorm, cmap='gray')
            ax7.imshow(hsvimgsnorm, cmap='Set3')
            ax7.set_title('HSV-S-Norm', fontsize=20)
            #ax8.imshow(hsvimgvnorm, cmap='gray')
            ax8.imshow(hsvimgvnorm, cmap='Set3')
            ax8.set_title('HSV-V-Norm', fontsize=20)

            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
            
            ax9.imshow(gradient, aspect='auto', cmap=plt.get_cmap('Set3'))
            ax9.set_title('Set3 Legend', fontsize=20)
            ticks = []
            for i in range(0,255,5):
                ticks.append(i)
            ax9.set_xticks(ticks, minor=True)
            
            ax10.set_axis_off()
            ax11.set_axis_off()
            ax12.set_axis_off()
            
            plt.show()
        
        # Use the chosen edge detection and color transformation for filtering the image
        filtered_image, combined_hls_filtered_image, white_filtered_image, yellow_filtered_image, \
        gradx_hls_l_filtered_image, gradx_hls_s_filtered_image, lab_l_filtered_image, lab_b_filtered_image  = self.grad_color_filter(birdimage,birdimagenorm)
        if self.deblev > 3:
            self.plotimg(filtered_image)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = filtered_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Transforming one dimensional, filtered and warped image into 3 channel image
        out_img = np.uint8(np.dstack((filtered_image, filtered_image, filtered_image))*255)
        white_filter_img = np.uint8(np.dstack((white_filtered_image, white_filtered_image, white_filtered_image))*255)
        yellow_filter_img = np.uint8(np.dstack((yellow_filtered_image, yellow_filtered_image, yellow_filtered_image))*255)
        combined_hls_filter_img = np.uint8(np.dstack((combined_hls_filtered_image, combined_hls_filtered_image, combined_hls_filtered_image))*255)
        gradx_hls_l_filter_img = np.uint8(np.dstack((gradx_hls_l_filtered_image, gradx_hls_l_filtered_image, gradx_hls_l_filtered_image))*255)
        gradx_hls_s_filter_img = np.uint8(np.dstack((gradx_hls_s_filtered_image, gradx_hls_s_filtered_image, gradx_hls_s_filtered_image))*255)
        lab_l_filter_img = np.uint8(np.dstack((lab_l_filtered_image, lab_l_filtered_image, lab_l_filtered_image))*255)
        lab_b_filter_img = np.uint8(np.dstack((lab_b_filtered_image, lab_b_filtered_image, lab_b_filtered_image))*255)

        ### Simulating first time lane detection based upon window methodology
        left_lane_inds, right_lane_inds, win_img = self.lineidx_detection_per_window(filtered_image, nonzerox, nonzeroy)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Combine image with windows and 3-dim bird image, print it out
        plt_img = cv2.addWeighted(out_img, 1, win_img, 1.0, 0)
        boxedpolyimg = self.plot_lanelines(plt_img, left_fit, right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy)


        ### Simulating second time lane detection based upon search area method
        nonzero = birdimage.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # lane detection based upon search area method (requiring fitted polynoms)    
        left_lane_inds, right_lane_inds, win_img = self.lineidx_detection_per_fit(filtered_image, nonzerox, nonzeroy, left_fit, right_fit)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Combine image with search area and 3-dim bird image, print it out
        plt_img = cv2.addWeighted(out_img, 1, win_img, 0.3, 0)
        contpolyimg = self.plot_lanelines(plt_img, left_fit, right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy)

        # Calculate curvature and offset from center of road for sanity check and print out into final picture
        left_curverad, right_curverad = self.calc_curvature(leftx, rightx, lefty, righty)
        xoffset = self.calc_xoffset(left_fit, right_fit, image.shape)
        sanity = self.sanity_check(left_fit, right_fit, left_fit, right_fit, 
                                   left_curverad, right_curverad, left_curverad, right_curverad, image.shape)

        # Unwarp bird view image, integrate the polynom lines as marker into the image
        markimage = self.gen_img_w_marks(image, filtered_image, left_fit, right_fit, Minv)

    ####
    #
    # Pic collection
    #
    ####

        if self.output == 'none':
            finimage = markimage
        elif self.output == 'info':
            # Add curvature, offset, and sanity information into final image
            finimage = markimage.copy();
            str =("left curve:   {:7.2f}m".format(left_curverad))
            cv2.putText(finimage, str, (100,40), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.2, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
            str =("right curve: {:7.2f}m".format(right_curverad))
            cv2.putText(finimage, str, (100,80), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.2, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
            str =("Center offset:  {:5.2f}m".format(xoffset))
            cv2.putText(finimage, str, (100,120), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.2, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
            str =("Sanity: {}".format(sanity))
            cv2.putText(finimage, str, (100,160), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.2, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
            str =("VP: ({},{})".format(vp[0][0],vp[1][0]))
            cv2.putText(finimage, str, (300,160), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.2, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)        
        elif self.output == 'small':
            finimage = self.plot_gallery_small(markimage, undistimage, birdimage, out_img, boxedpolyimg, contpolyimg, \
                                               left_curverad, right_curverad, xoffset, sanity, vp)

        elif self.output == 'medium':
            finimage = self.plot_gallery_medium(markimage, undistimage, birdimage, out_img, boxedpolyimg, contpolyimg, \
                                                combined_hls_filter_img, yellow_filter_img, \
                                                left_curverad, right_curverad, xoffset, sanity, vp)

        elif self.output == 'deluxe':        
            finimage = self.plot_galleray_deluxe(markimage, undistimage, birdimage, out_img, boxedpolyimg, contpolyimg, \
                                                 combined_hls_filter_img, yellow_filter_img, \
                                                 lab_l_filter_img, lab_b_filter_img, gradx_hls_l_filter_img, gradx_hls_s_filter_img, white_filter_img, \
                                                 left_curverad, right_curverad, xoffset, sanity, vp)
        else:
            print("ERROR: Unknown output \'%s\'!" % self.output)
            exit()

        return finimage
    
    def process_movie(self, image):
        """
        Function for processing consecutive images of a movie
        
        Input: RGB Image 'image'
        Output: Image with masked lane area and curvature, offset information ('marked_image')
        """
        
        sanity = 1
        vpchanged = 0
        
        # Chosen camera matrix and distortion values
        mtx = np.array([[  1.15396093e+03,   0.00000000e+00,   6.69705359e+02],
               [  0.00000000e+00,   1.14802495e+03,   3.85656232e+02],
               [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
        dist = np.array([[ -2.41017968e-01,  -5.30720497e-02,  -1.15810318e-03,  -1.28318543e-04, 2.67124302e-02]])

        # Undistorting image
        undistimage = cv2.undistort(image, mtx, dist, None, mtx)

        hsvimgnorm  = self.convert_image(undistimage, method='rgb2hsvnorm')
        rgbimgnorm = cv2.cvtColor(hsvimgnorm, cv2.COLOR_HSV2RGB)

        # Warping image
        birdimagenorm, perspective_M, Minv, vp  = self.corners_unwarp(rgbimgnorm)
        birdimage, perspective_M, Minv, vp = self.corners_unwarp(undistimage)

        vpchanged = 1
        #self.vp = vp
        #self.Minv = Minv
        
       
        # Color transformation and filtering according to best method evaluated
        filtered_image, combined_hls_filtered_image, white_filtered_image, yellow_filtered_image, \
        gradx_hls_l_filtered_image, gradx_hls_s_filtered_image, lab_l_filtered_image, lab_b_filtered_image  = self.grad_color_filter(birdimage, birdimagenorm)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = filtered_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Transforming one-dimensional birds view image into 3-dim image
        out_img = np.uint8(np.dstack((filtered_image, filtered_image, filtered_image))*255)
        white_filter_img = np.uint8(np.dstack((white_filtered_image, white_filtered_image, white_filtered_image))*255)
        yellow_filter_img = np.uint8(np.dstack((yellow_filtered_image, yellow_filtered_image, yellow_filtered_image))*255)
        combined_hls_filter_img = np.uint8(np.dstack((combined_hls_filtered_image, combined_hls_filtered_image, combined_hls_filtered_image))*255)
        gradx_hls_l_filter_img = np.uint8(np.dstack((gradx_hls_l_filtered_image, gradx_hls_l_filtered_image, gradx_hls_l_filtered_image))*255)
        gradx_hls_s_filter_img = np.uint8(np.dstack((gradx_hls_s_filtered_image, gradx_hls_s_filtered_image, gradx_hls_s_filtered_image))*255)
        lab_l_filter_img = np.uint8(np.dstack((lab_l_filtered_image, lab_l_filtered_image, lab_l_filtered_image))*255)
        lab_b_filter_img = np.uint8(np.dstack((lab_b_filtered_image, lab_b_filtered_image, lab_b_filtered_image))*255)
        
        ### Simulating first time lane detection based upon window methodology
        left_lane_inds, right_lane_inds, win_img = self.lineidx_detection_per_window(filtered_image, nonzerox, nonzeroy)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Combine image with windows and 3-dim bird image, print it out
        plt_img = cv2.addWeighted(out_img, 1, win_img, 1.0, 0)
        boxedpolyimg = self.plot_lanelines(plt_img, left_fit, right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy)


        ### Simulating second time lane detection based upon search area method
        nonzero = birdimage.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # lane detection based upon search area method (requiring fitted polynoms)    
        left_lane_inds, right_lane_inds, win_img = self.lineidx_detection_per_fit(filtered_image, nonzerox, nonzeroy, left_fit, right_fit)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Combine image with search area and 3-dim bird image, print it out
        plt_img = cv2.addWeighted(out_img, 1, win_img, 0.3, 0)
        contpolyimg = self.plot_lanelines(plt_img, left_fit, right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy)

        # Choosing either window method (first picture) or search area (all other pictures) for lane line detection
        if self.first_pic_done:
            self.left_curverad_old, self.right_curverad_old = self.left_curverad, self.right_curverad
            self.left_curverad, self.right_curverad = self.calc_curvature(leftx, rightx, lefty, righty)
            sanity = self.sanity_check_after_rev(left_fit, right_fit, self.left_fit_glob, self.right_fit_glob, 
                                            self.left_curverad, self.right_curverad, self.left_curverad_old, self.right_curverad_old, image.shape)
            if sanity:
                self.left_fit_glob = left_fit
                self.right_fit_glob = right_fit
                self.nosanitycnt = 0
            else:
                self.nosanitycnt += 1
                if (self.nosanitycnt >= self.sancntthres):
                    self.nosanitycnt = 0
                    left_lane_inds, right_lane_inds, win_img = self.lineidx_detection_per_window(filtered_image, nonzerox, nonzeroy)
                    # Extract left and right line pixel positions
                    leftx = nonzerox[left_lane_inds]
                    lefty = nonzeroy[left_lane_inds] 
                    rightx = nonzerox[right_lane_inds]
                    righty = nonzeroy[right_lane_inds] 

                    # Fit a second order polynomial to each
                    self.left_fit_glob = np.polyfit(lefty, leftx, 2)
                    self.right_fit_glob = np.polyfit(righty, rightx, 2)

                    self.left_curverad, self.right_curverad = self.calc_curvature(leftx, rightx, lefty, righty)        
        else:
            self.left_fit_glob = left_fit
            self.right_fit_glob = right_fit

            self.left_curverad, self.right_curverad = self.calc_curvature(leftx, rightx, lefty, righty)
            first_pic_done = 1

        xoffset = self.calc_xoffset(self.left_fit_glob, self.right_fit_glob, image.shape)
       
        # Unwarp bird view image, integrate the polynom lines as marker into the image
        markimage = self.gen_img_w_marks(image, filtered_image, self.left_fit_glob, self.right_fit_glob, Minv)

        ####
        #
        # Pic collection
        #
        ####

        if self.output == 'none':
            finimage = markimage
        elif self.output == 'info':
            # Add curvature, offset, and sanity information into final image
            finimage = markimage.copy();
            str =("left curve:   {:7.2f}m".format(self.left_curverad))
            cv2.putText(finimage, str, (100,40), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.2, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
            str =("right curve: {:7.2f}m".format(self.right_curverad))
            cv2.putText(finimage, str, (100,80), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.2, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
            str =("Center offset:  {:5.2f}m".format(xoffset))
            cv2.putText(finimage, str, (100,120), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.2, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
            str =("Sanity: {}".format(sanity))
            cv2.putText(finimage, str, (100,160), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.2, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
            str =("VP: ({},{})".format(vp[0][0],vp[1][0]))
            cv2.putText(finimage, str, (300,160), cv2.FONT_HERSHEY_TRIPLEX, fontScale=1.2, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)        
        elif self.output == 'small':
            finimage = self.plot_gallery_small(markimage, undistimage, birdimage, out_img, boxedpolyimg, contpolyimg, \
                                               self.left_curverad, self.right_curverad, xoffset, sanity, vp)

        elif self.output == 'medium':
            finimage = self.plot_gallery_medium(markimage, undistimage, birdimage, out_img, boxedpolyimg, contpolyimg, \
                                                combined_hls_filter_img, yellow_filter_img, \
                                                self.left_curverad, self.right_curverad, xoffset, sanity, vp)

        elif self.output == 'deluxe':        
            finimage = self.plot_galleray_deluxe(markimage, undistimage, birdimage, out_img, boxedpolyimg, contpolyimg, \
                                                 combined_hls_filter_img, yellow_filter_img, \
                                                 lab_l_filter_img, lab_b_filter_img, gradx_hls_l_filter_img, gradx_hls_s_filter_img, white_filter_img, \
                                                 self.left_curverad, self.right_curverad, xoffset, sanity, vp)
        else:
            print("ERROR: Unknown output \'%s\'!" % self.output)
            exit()

        return finimage
    
    def run (self, media):
        if self.deblev > 0:
            begtime = datetime.now()
        
        if self.mode == 'picture':
            if self.deblev > 0:
                print('Picture mode')
            mediaobj = self.process_single_image(media)
        elif self.mode == 'video':
            if self.deblev > 0:
                print('Video mode')
            movie = VideoFileClip(media)
            mediaobj = movie.fl_image(self.process_movie)
        else:
            print("ERROR: Unknown mode \'%s\'!" % self.mode)
            exit()
 
        if self.deblev > 0:
            endtime = datetime.now()
            print("Start: ", begtime, "\nEnd:   ", endtime)
            duration = endtime - begtime
            print("Duration: ", duration, "\n" )
            
        return mediaobj
