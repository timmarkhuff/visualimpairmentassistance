# Import packages
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import pyttsx3
import math
from re import X

def get_destination_points(corners):
    """
    -Get destination points from corners of warped images
    -Approximating height and width of the rectangle: we take maximum of the 2 widths and 2 heights

    Args:
        corners: list

    Returns:
        destination_corners: list
        height: int
        width: int

    """

    w1 = np.sqrt((corners[0][0] - corners[1][0]) ** 2 + (corners[0][1] - corners[1][1]) ** 2)
    w2 = np.sqrt((corners[2][0] - corners[3][0]) ** 2 + (corners[2][1] - corners[3][1]) ** 2)
    w = max(int(w1), int(w2))

    h1 = np.sqrt((corners[0][0] - corners[2][0]) ** 2 + (corners[0][1] - corners[2][1]) ** 2)
    h2 = np.sqrt((corners[1][0] - corners[3][0]) ** 2 + (corners[1][1] - corners[3][1]) ** 2)
    h = max(int(h1), int(h2))

    destination_corners = np.float32([(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)])


    # print('\nThe destination points are: \n')
    
    for index, c in enumerate(destination_corners):
        character = chr(65 + index) + "'"
        # print(character, ':', c)

    # print('\nThe approximated height and width of the original image is: \n', (h, w))
    return destination_corners, h, w


def unwarp(img, src, dst, plotting_mode=0):
    """

    Args:
        img: np.array
        src: list
        dst: list

    Returns:
        un_warped: np.array

    """
    h, w = img.shape[:2]
    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    # print('\nThe homography matrix is: \n', H)
    un_warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)

    if plotting_mode:
      f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
      f.subplots_adjust(hspace=.2, wspace=.05)
      ax1.imshow(img)
      ax1.set_title('Original Image')

      x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
      y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]

      ax2.imshow(img)
      ax2.plot(x, y, color='yellow', linewidth=3)
      ax2.set_ylim([h, 0])
      ax2.set_xlim([0, w])
      ax2.set_title('Target Area')

    return un_warped

def apply_filter(image, plotting_mode=0):
    """
    Define a 5X5 kernel and apply the filter to gray scale image
    Args:
        image: np.array

    Returns:
        filtered: np.array

    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((5, 5), np.float32) / 15
    filtered = cv2.filter2D(gray, -1, kernel)
    
    if plotting_mode:
      # plot
      plt.imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
      plt.title('Filtered Image')
      plt.show()

    return filtered

def apply_threshold(filtered, plotting_mode=0):
    """
    Apply OTSU threshold

    Args:
        filtered: np.array

    Returns:
        thresh: np.array

    """
    ret, thresh = cv2.threshold(filtered, 250, 255, cv2.THRESH_OTSU)

    thresh = cv2.bitwise_not(thresh)
    
    if plotting_mode:
      # plot
      plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
      plt.title('After applying OTSU threshold')
      plt.show()

    return thresh

def detect_contour(img, image_shape, plotting_mode=0):
    """

    Args:
        img: np.array()
        image_shape: tuple

    Returns:
        canvas: np.array()
        cnt: list

    """
    canvas = np.zeros(image_shape, np.uint8)

    ####
    try:
      # this is the required syntax for the Raspberry Pi
      _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    except:
      # this is the required syntax for Google Colab
      contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    cv2.drawContours(canvas, cnt, -1, (0, 255, 255), 3)
    
    if plotting_mode:
      # plot
      plt.title('Largest Contour')
      plt.imshow(canvas)
      plt.show()

    return canvas, cnt

def detect_corners_from_contour(canvas, cnt, plotting_mode=0):
    """
    Detecting corner points form contours using cv2.approxPolyDP()
    Args:
        canvas: np.array()
        cnt: list

    Returns:
        approx_corners: list

    """
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx_corners = cv2.approxPolyDP(cnt, epsilon, True)
    cv2.drawContours(canvas, approx_corners, -1, (255, 255, 0), 10)
    approx_corners = sorted(np.concatenate(approx_corners).tolist())
    # print('\nThe corner points are ...\n')
    for index, c in enumerate(approx_corners):
        character = chr(65 + index)
        # print(character, ':', c)
        cv2.putText(canvas, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    #############################################
    # TIM'S METHOD FOR CORRECTLY ORDERING CORNERS
    # create a large bounding rectangle to calcuate closest corners
    canvas_x = float(canvas.shape[1])
    canvas_y = float(canvas.shape[0])
    bounding_corners = [[0., 0.], [canvas_x, 0.], [0., canvas_y], [canvas_x, canvas_y]] 

    # print(f'bounding_corners: {bounding_corners}')

    # corners = approx_corners
    closest_corner_list = []
    for bounding_corner in bounding_corners:
      curr_dist = 100000
      closest_corner = ''
      dest_x = bounding_corner[0]
      dest_y = bounding_corner[1]
      for corner in approx_corners:
        x = corner[0]
        y = corner[1]
        dist = abs(math.hypot(dest_x - x, dest_y - y))
        # print(f"distance from {corner} to bounding corner {bounding_corner} is {dist}")
        if dist < curr_dist:
          curr_dist = dist
          closest_corner = corner
      closest_corner_list.append(closest_corner)
    
    approx_corners = closest_corner_list
    # print(f"here are the final corners: {approx_corners}")
    #############################################

    # #####################################################################
    # # ORIGINAL METHOD FOR REARRANGING CORNERS
    # approx_corners = [approx_corners[i] for i in [0, 2, 1, 3]]
    # #####################################################################

    if plotting_mode:
      # plot
      plt.imshow(canvas)
      plt.title('Corner Points: Douglas-Peucker')
      plt.show()

    return approx_corners, canvas

def hsv_threshold(img, plotting_mode=0):
    """
    for identifying the rectangular shape of the sign
    for real safeway signs rather than test signs
    returns a mask that can be used for deskewing
    """

    #convert the BGR image to HSV colour space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #set the lower and upper bounds for target color
    lower = np.array([70,30,0])
    upper = np.array([120,255,200])

    #create a mask based on an HSV range
    mask = cv2.inRange(hsv, lower, upper)

    # # Inverting the mask 
    # mask = cv2.bitwise_not(mask)

    # #perform bitwise and on the original image arrays using the mask # don't think this is needed
    # thresh = cv2.bitwise_and(image, image, mask=mask)

    if plotting_mode:
      plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
      plt.title("After applying Tim's mask")
      plt.show()

    return mask

def process_and_unwarp(image, test_mode=0, plotting_mode=0):
    """
    Skew correction using homography and corner detection using contour points
    test_mode = 0: for fake signs
    test_mode = 1: for real Safeway signs
    Returns: an unwarped image: numpy array

    """
    
    if plotting_mode:
      # plot
      plt.imshow(image)
      plt.title('Original Image')
      plt.show()

    if test_mode:
      # original method - for fake signs 
      rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      filtered_image = apply_filter(rgb_image, plotting_mode=plotting_mode)
      mask = apply_threshold(filtered_image, plotting_mode=plotting_mode)
    else:
      # for real signs
      mask = hsv_threshold(image, plotting_mode=plotting_mode)

    # find countours and corners
    cnv, largest_contour = detect_contour(mask, image.shape, plotting_mode=plotting_mode)
    corners, canvas = detect_corners_from_contour(cnv, largest_contour, plotting_mode=plotting_mode)

    _, h, w = get_destination_points(corners)

    offset_x = (image.shape[1] - w) / 2
    offset_y = (image.shape[0] - h) / 2

    destination_points = np.float32([(offset_x, offset_y), (offset_x + w, offset_y), (offset_x, offset_y + h), (offset_x + w, offset_y + h)])

    un_warped = unwarp(image, np.float32(corners), destination_points, plotting_mode=plotting_mode)

    cropped = un_warped[0:h, 0:w]
    
    if plotting_mode:
      # plot
      f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
      # f.subplots_adjust(hspace=.2, wspace=.05)
      ax1.imshow(un_warped)
      # ax2.imshow(cropped)
      plt.show()

    # convert binary mask to RGB so that it can be concatenated with other images
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # concatenate unwarping images
    top = cv2.hconcat([image, mask_rgb])
    bottom = cv2.hconcat([canvas, un_warped])
    unwarp_process_image = cv2.vconcat([top, bottom])

    return un_warped, unwarp_process_image

def fig2img(fig):
  fig.canvas.draw()
  image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  return image_from_plot

def darius_ocr_v2(img):
  """
  -takes an image
  -processes the image
  -performs OCR on the processed image
  -returns mask and dictionary
  """
  img = cv2.resize(img, (0, 0), fx=3, fy=3)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  kernel = np.ones((2,2),np.uint8)
  erosion = cv2.erode(gray,kernel,iterations = 2)
  mask = cv2.bitwise_not(erosion)
  ocr_dict = pytesseract.image_to_data(mask, lang='eng', config='--psm 11', output_type=Output.DICT)

  return mask, ocr_dict

def parse_ocr_dict(ocr_dict, conf=50):
  final_text = ""

  for index, i in enumerate(ocr_dict['text']):
    # remove all non alphanumeric characters from the string
    cleaned_string = ''.join(filter(str.isalnum, i))

    # filter out words with a low confidence score
    if int(ocr_dict['conf'][index]) > conf:
      final_text = final_text + cleaned_string
      curr_block_num = int(ocr_dict['block_num'][index])
      try:
          next_block_num = int(ocr_dict['block_num'][index + 1])
      except:
          next_block_num = int(ocr_dict['block_num'][index]) + 1
      if next_block_num != curr_block_num:

        # add a semicolon to the end if this is the last word on the block
        final_text = final_text + "; "
      else:
        # add a space to the end of the string if this is not the last word 
        # on the block
        final_text = final_text + " "

  return final_text



def ocr_darius(img):

  img = cv2.resize(img, (0, 0), fx=3, fy=3)
  kernel = np.ones((2,2),np.uint8)
  erosion = cv2.erode(img,kernel,iterations = 2)
  mask = erosion

  # OCR
#   custom_config = r'--oem 3 --psm 6 outputbase digits'
#   txt = pytesseract.image_to_string(mask, config=custom_config)
  txt = pytesseract.image_to_string(mask, lang='eng')
  
  # clean the string
  txt = ''.join(c for c in txt if c.isalnum() or c == " " or c == "\n") # remove non alphanuemeric characters
  txt = txt.replace('\n', ', ')
      
  return mask, txt


def ocr(img):
    """
    -takes an image
    -processes the image
    -performs OCR on the processed image
    -returns mask and text
    """
    
    # BEGIN TESSERACT
    # Up-sample
    img = cv2.resize(img, (0, 0), fx=2, fy=2)

    # Convert to greyscale
    # greyscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to HSV color-space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
 
    # Get the binary mask
    # mask = cv2.inRange(greyscale, 0, 62)
    mask = cv2.inRange(hsv, np.array([0, 0, 90]), np.array([179, 255, 255]))
    mask = cv2.inRange(hsv, np.array([0, 0, 60]), np.array([179, 255, 255]))

    #invert the binary mask
    mask = cv2.bitwise_not(mask)

#     kernel = np.ones((2,2), np.uint8)
#     mask = cv2.erode(img, kernel, iterations = 4)

    # OCR
    txt = pytesseract.image_to_string(mask, lang='helvetica_bold.traineddata')
    
    # clean the string
    txt = ''.join(c for c in txt if c.isalnum() or c == " " or c == "\n") # remove non alphanuemeric characters
    txt = txt.replace('\n', ' ')
        
    return mask, txt

