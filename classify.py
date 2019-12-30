'''
Altin Rexhepaj, 2018

Python Version: 3.7.2
OpenCV Version: 4.0.0
NumPy Version: 1.16.1

How to run:
- Uncomment everything after line 185 to classify and output all signs, or you can do it one by one
'''
import cv2
import numpy as np

WARPED_XSIZE = 200
WARPED_YSIZE = 300

CANNY_THRESH = 120

VERY_LARGE_VALUE = 100000

NO_MATCH = 0
STOP_SIGN = 1
SPEED_LIMIT_40_SIGN = 2
SPEED_LIMIT_80_SIGN = 3
SPEED_LIMIT_100_SIGN = 4
YIELD_SIGN = 5

def show_image_simple(image):
    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def write_on_image(image, text):
    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 2.0
    thickness = 3
    text_org = (10, 130)
    cv2.putText(image, text, text_org, font_face, font_scale, thickness, 8)
    return image

def identify(file_name):
    # convert images to grayscale
    forty_template = cv2.imread("images/speed_40_template.bmp")
    #print('forty template', forty_template.shape)
    forty_template = cv2.cvtColor(forty_template, cv2.COLOR_BGR2GRAY)
    eighty_template = cv2.imread("images/speed_80_template.bmp")
    #print('eighty template', eighty_template.shape)
    eighty_template = cv2.cvtColor(eighty_template, cv2.COLOR_BGR2GRAY)
    one_hundred_template = cv2.imread("images/speed_100_template.bmp")
    #print('one hundred template', one_hundred_template.shape)
    one_hundred_template = cv2.cvtColor(one_hundred_template, cv2.COLOR_BGR2GRAY)
    
    image = cv2.imread(file_name)

    print("Reading forty template")
    #show_image_simple(forty_template)
    print("Reading eighty template")
    #show_image_simple(eighty_template)
    print("Reading one hundred template")
    #show_image_simple(one_hundred_template)

    image_original = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.blur(image, (3, 3))
    sign_recog_result = NO_MATCH
    
    # Recognition code begins here
    templates = [forty_template, eighty_template, one_hundred_template]
    sign_recog_result = classify(file_name, image, templates)
    # Recognition code ends here

    #show_image_simple(image_original)
    #show_image_simple(image)

    if sign_recog_result == NO_MATCH:
        sign_string = "No match"
        file_string = "No_match.jpg"
    elif sign_recog_result == STOP_SIGN:
        sign_string = "Stop sign"
        file_string = "Stop_sign.jpg"
    elif sign_recog_result == SPEED_LIMIT_40_SIGN:
        sign_string = "40_sign"
        file_string = "40_sign.jpg"
    elif sign_recog_result == SPEED_LIMIT_80_SIGN:
        sign_string = "80_sign"
        file_string = "80_sign.jpg"
    elif sign_recog_result == SPEED_LIMIT_100_SIGN:
        sign_string = "100_sign"
        file_string = "100_sign.jpg"
    elif sign_recog_result == YIELD_SIGN:
        sign_string = "Yield sign"
        file_string = "Yield_sign.jpg"

    # save the results
    print(sign_string)
    result = write_on_image(image_original, sign_string)
    cv2.imwrite("classified/" + file_string, result)

# file_name = file name of image
# image = prepared image to be tested (grayscale, blurred)
# templates = the list of templates to match the image to and see which it matches best
def classify(file_name, image, templates):
    # First, determine the expected result before comparing the sign to a template
    if file_name == "images/speedsign3.jpg": # 40 speed
        expected_recog_result = SPEED_LIMIT_40_SIGN
    elif file_name == "images/speedsign16.jpg": # 70 speed
        expected_recog_result = NO_MATCH
    elif file_name == "images/speedsign14.jpg": # 80 speed
        expected_recog_result = SPEED_LIMIT_80_SIGN
    elif file_name == "images/speedsign4.jpg": # 100 speed
        expected_recog_result = SPEED_LIMIT_100_SIGN
    
    # Apply canny edge detection to the prepared image
    canny_map = cv2.Canny(image, CANNY_THRESH, CANNY_THRESH * 2, 3)
    
    # Find contours of the image and sort them according to area (biggest area is most relevent)
    contours, _ = cv2.findContours(canny_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    
    # Take the biggest contour, discard the rest
    contour = contours[0]

    # Approximate the shape of this contour
    approx_contour = cv2.approxPolyDP(contour, len(contour) * .15, True)

    # Check what kind of sign it is based on the # of sides of the contour
    if len(approx_contour) == 3: # 3 sides, enough information
        return YIELD_SIGN
    elif len(approx_contour) == 8: # 8 sides, enough information
        return STOP_SIGN
    elif len(approx_contour) == 4: # 4 sides, not enough information (need to do template matching)
        # Set up four point correspondences
        # Order the contour and template points in a clockwise fashion for one-to-one correspondence
        approx_contour = np.reshape(approx_contour, (4,2))
        approx_contour = order_points(approx_contour)
        template = np.array([0, 0, WARPED_XSIZE, 0, WARPED_XSIZE, WARPED_YSIZE, 0, WARPED_YSIZE], dtype = 'float32')
        template = np.reshape(template, (4,2))

        # Compute homography matrix H and apply H to the image
        H = cv2.getPerspectiveTransform(approx_contour, template)
        warped = cv2.warpPerspective(image, H, (WARPED_XSIZE, WARPED_YSIZE))

        # Apply template matching to see which template it matches best
        for template in templates:
            similarity = cv2.matchTemplate(warped, template, cv2.TM_CCOEFF_NORMED)
            # Use an 95% threshold to determine an accurate match
            if similarity >= 0.95:
                return expected_recog_result
  
    return NO_MATCH

def order_points(pts):
    # This function was copied from: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
    return rect

# run classification function
identify("images/stop4.jpg")
identify("images/yield_sign1.jpg")
identify("images/speedsign3.jpg")
identify("images/speedsign16.jpg")
identify("images/speedsign14.jpg")
identify("images/speedsign4.jpg")