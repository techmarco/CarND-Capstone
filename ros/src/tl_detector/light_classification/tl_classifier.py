import numpy as np
import rospy
import cv2
from styx_msgs.msg import TrafficLight


class TLClassifier(object):
    def __init__(self):
        self.light_color = None

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        img_h, img_w = image.shape[0], image.shape[1]
        real = np.copy(image)
        orig = np.copy(image)

        kernel_close = np.ones((3, 3), np.uint8)
        kernel_close2 = np.ones((5, 3), np.uint8)
        kernel_far = np.ones((11, 3), np.uint8)

        image = cv2.blur(image, (3, 3))
        imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        imgray = cv2.normalize(imgray, imgray, 0, 255, cv2.NORM_MINMAX)

        dil_image_close = cv2.erode(imgray, kernel_close, iterations=1)
        dil_image_close2 = cv2.erode(imgray, kernel_close2, iterations=1)
        dil_image_far = cv2.erode(imgray, kernel_far, iterations=1)

        dil_image_dilate = cv2.dilate(imgray, kernel_close, iterations=1)
        dil_image_dilate2 = cv2.dilate(imgray, kernel_close2, iterations=1)


        _, thresh_close = cv2.threshold(dil_image_close, 20, 160, cv2.THRESH_BINARY)
        _, thresh_close2 = cv2.threshold(dil_image_close2, 6, 160, cv2.THRESH_BINARY)
        _, thresh_far = cv2.threshold(dil_image_far, 3, 160, cv2.THRESH_BINARY)

        _, thresh_close_sans = cv2.threshold(dil_image_dilate, 24, 160, cv2.THRESH_BINARY)
        _, thresh_close2_sans = cv2.threshold(dil_image_dilate2, 20, 160, cv2.THRESH_BINARY)


        edges = cv2.Canny(thresh_close2, 40, 80, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180., 40, minLineLength=20, maxLineGap=40)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if (y2-y1)**2 > (x2-x1)**2  and abs(x2-x1) < 6 and abs(y2-y1) < img_h / 3:
                    if y2 > y1:
                        cv2.line(thresh_close2, (x1, y1), (x2, y2), (160), 4)
                    else:
                        cv2.line(thresh_close2, (x1, y1), (x2, y2), (160), 4)

        edges = cv2.Canny(thresh_close, 40, 80, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180., 40, minLineLength=20, maxLineGap=40)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if (y2-y1)**2 > (x2-x1)**2 and abs(x2-x1) < 6 and abs(y2-y1) < img_h / 3:
                    if y2 > y1:
                        cv2.line(thresh_close, (x1, y1), (x2, y2), (160), 4)
                    else:
                        cv2.line(thresh_close, (x1, y1), (x2, y2), (160), 4)

        edges = cv2.Canny(thresh_far, 40, 80, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180., 40, minLineLength=20, maxLineGap=40)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2-y1)**2 > (x2-x1)**2 and abs(x2-x1) < 6 and abs(y2-y1) < img_h / 3:
                    if y2 > y1:
                        cv2.line(thresh_far, (x1, y1), (x2, y2), (160), 4)
                    else:
                        cv2.line(thresh_far, (x1, y1), (x2, y2), (160), 4)


        _, cntrs_close, _ = cv2.findContours(thresh_close, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        _, cntrs_close2, _ = cv2.findContours(thresh_close2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        _, cntrs_far, _ = cv2.findContours(thresh_far, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        _, cntrs_closeA, _ = cv2.findContours(thresh_close_sans, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        _, cntrs_closeB, _ = cv2.findContours(thresh_close2_sans, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)


        red_light_votes = 0
        yellow_light_votes = 0
        green_light_votes = 0
        unknown_votes = 0

        cntrs = cntrs_close + cntrs_close2 + cntrs_far + cntrs_closeA + cntrs_closeB

        font = cv2.FONT_HERSHEY_SIMPLEX
        kept_cntrs = []
        idx = -1
        for c in cntrs:
            idx += 1

            area = cv2.contourArea(c)
            if area < 2000 or area > 15000:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.05*peri, True)

            (x, y, w, h) = cv2.boundingRect(c)
            aspect = w / float(h)
            if aspect > 0.3 and aspect < 0.6:
                color = (255, 255, 255)
                if idx >= len(cntrs_close) and idx < len(cntrs_close) + len(cntrs_close2):
                    y_off = -5
                    x_off = w + 5
                    color = (255, 0, 0)
                elif idx >= len(cntrs_close) + len(cntrs_close2):
                    y_off = h/2
                    x_off = w + 5
                    color = (0, 255, 0)
                else:
                    y_off = h + 5
                    x_off = w + 5
                    color = (0, 0, 255)

                if w > 6:
                    x += int(0.1 * w)
                    w -= int(0.2 * w)
                if h > 6:
                    y += int(0.1 * h)
                    h -= int(0.2 * h)

                cv2.rectangle(orig, (x,y), (x+w,y+h), color, 3)

                top_sum = real[y:y+h/3, x:x+w, :].sum(axis=2)

                blue_top = real[y:y+h/3, x:x+w, 0].sum()
                green_top = real[y:y+h/3, x:x+w, 1].sum()
                red_top = real[y:y+h/3, x:x+w, 2].sum()

                blue_mid = real[y+h/3:y+2*h/3, x:x+w, 0].sum()
                green_mid = real[y+h/3:y+2*h/3, x:x+w, 1].sum()
                red_mid = real[y+h/3:y+2*h/3, x:x+w, 2].sum()

                blue_bot = real[y+2*h/3:y+h, x:x+w, 0].sum()
                green_bot = real[y+2*h/3:y+h, x:x+w, 1].sum()
                red_bot = real[y+2*h/3:y+h, x:x+w, 2].sum()

                total_top = np.sum([blue_top, green_top, red_top])
                total_mid = np.sum([blue_mid, green_mid, red_mid])
                total_bot = np.sum([blue_bot, green_bot, red_bot])

                weight = 1.75

                light = 'unknown'
                color = (255, 255, 255)
                if (red_mid + green_mid) > 1.25*weight*(red_top + green_top) and (red_mid + green_mid) > 1.25*weight*(red_bot + green_bot):
                    light = 'yellow'
                    color = (0, 255, 255)
                    yellow_light_votes += 1
                elif red_top > weight*red_mid and red_top > weight*red_bot:
                    light = 'red'
                    color = (0, 0, 255)
                    red_light_votes += 1
                elif green_bot > 1.25*weight*green_top and green_bot > 1.25*weight*green_mid:
                    light = 'green'
                    color = (0, 255, 0)
                    green_light_votes += 1
                else:
                    unknown_votes += 1

                cv2.putText(orig, light, (x+x_off,y+y_off), font, 1, color, 2, cv2.LINE_AA)

        # cv2.imshow('far', thresh_far)
        # cv2.moveWindow('far', 1200, 0)
        # cv2.imshow('thresh_close', thresh_close_sans)
        # cv2.moveWindow('thresh_close', 1200, 600)
        # cv2.imshow('close2', thresh_close2_sans)
        # cv2.moveWindow('close2', 100, 600)
        cv2.imshow('image', orig)
        cv2.moveWindow('image', 320, 130)
        cv2.waitKey(1)

        if green_light_votes > 0 and green_light_votes > (yellow_light_votes + red_light_votes):
            if self.light_color != 'green':
                rospy.logwarn('----------------------------------->>>> GREEN')
                self.light_color = 'green'
            return TrafficLight.GREEN
        elif yellow_light_votes > 0 and yellow_light_votes > (green_light_votes + red_light_votes):
            # Slow for Yellow lights
            if self.light_color != 'yellow':
                rospy.logwarn('----------------------------------->>>> YELLOW')
                self.light_color = 'yellow'
            # return TrafficLight.YELLOW
            return TrafficLight.RED
        elif red_light_votes > 0 and red_light_votes > (green_light_votes + yellow_light_votes):
            if self.light_color != 'red':
                rospy.logwarn('----------------------------------->>>> RED')
                self.light_color = 'red'
            return TrafficLight.RED
        else:
            rospy.logwarn('----------------------------------->>>> UNKNOWN')
            return TrafficLight.UNKNOWN
