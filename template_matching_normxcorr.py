#!/usr/bin/env python3
import cv2
import numpy as np
import sys

# Compute normalized cross correlation scores
# which measure the "similarity" between
# a given template and a search image
def normxcorr2(template, image):

  # Inputs
  # template: grayscale template image (2D float array)
  # image: grayscale search image (2D float array)

  template_gray = template
  image_gray = image

  h_temp, w_temp = template_gray.shape
  h_img, w_img = image_gray.shape

  min_temp = 0
  max_temp = 0
  for i in range(0, h_temp):
    for j in range(0, w_temp):
      if template_gray[i][j] > max_temp:
        max_temp = template_gray[i][j]
      if template_gray[i][j] < min_temp:
        min_temp = template_gray[i][j]

  min_img = 0
  max_img = 0
  for i in range(0, h_img):
    for j in range(0, w_img):
      if image_gray[i][j] > max_img:
        max_img = image_gray[i][j]
      if image_gray[i][j] < min_temp:
        min_temp = image_gray[i][j]

  # Normalize individual pixel values
  template_norm = (template_gray - min_temp) / (max_temp - min_temp)
  image_norm = (image_gray - min_img) / (max_img - min_img)

  # Returns a matrix of normalized cross correlation scores
  ncc_matrix = np.zeros((h_img-h_temp+1, w_img-w_temp+1))

  for row in range(0, h_img-h_temp+1):
    for col in range(0, w_img-w_temp+1):

      # Compare given template with
      # a subimage of the same size
      template = template_norm
      sub_image = image_norm[row:row+h_temp, col:col+w_temp]

      # Cross correlate template and subimage
      # using formulas
      correlation = np.sum(template*sub_image)
      normal = np.sqrt( np.sum(template**2) ) * np.sqrt( np.sum(sub_image**2))
      score = correlation / normal
      ncc_matrix[row,col] = score

  return ncc_matrix

# Find the coordinates of the matches
# given a template and a search image
def find_matches(template, image, thresh=None):

  # Inputs
  # template: BGR template image (3D uint8 array)
  # image: BGR search image (3D uint8 array)

  template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
  image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Obtain normalized cross correlation matrix and
  # initialize other relevant variables
  corr_matrix = normxcorr2(template_gray, image_gray)
  image_copy = image_gray
  coordinates = []
  match_images = []
  max_val = 0
  (max_y, max_x) = (0, 0)

  h_temp, w_temp = template_gray.shape
  h_img, w_img = image_gray.shape

  color = (0,255,0)
  thickness = 2

  if thresh:
    max_val = thresh
  else:
    max_val = 0

  # Loop through ncc matrix
  for row in range(0, h_img-h_temp+1):
    for col in range(0, w_img-w_temp+1):

      # If threshold specified, return list of coordinates whose
      # values > threshold
      if thresh:
        if corr_matrix[row,col] > thresh:
          (max_y, max_x) = row, col
          coordinates.append((max_x,max_y))
          start_point = (max_x,max_y)
          end_point = (max_x+w_temp, max_y+h_temp)
          image = cv2.rectangle(image, start_point, end_point, color, thickness)
          match_images.append(image)
          image_gray = image_copy
      else:
        if corr_matrix[row,col] > max_val:
          (max_y, max_x) = row, col
          max_val = corr_matrix[row,col]

  # If threshold not specified by user, only return max coordinates
  if thresh == None:
    start_point = (max_x, max_y)
    end_point = (max_x + w_temp, max_y + h_temp)
    coords = (max_x, max_y)
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
    return coords, image

  return coordinates, match_images # return the last image


# Driver function
def main(argv):
  template_name = argv[0]
  image_name = argv[1]
  thresh = None
  try:
    thresh = argv[2]
  except:
    pass
  image = cv2.imread('data/' + image_name + '.png', cv2.IMREAD_COLOR)
  template = cv2.imread('data/' + template_name + '.png', cv2.IMREAD_COLOR)

  if thresh == None:
    coords, match_image = find_matches(template, image)
    cv2.imwrite('output/' + image_name + '.png', match_image)
  else:
    thresh = float(thresh)
    coords, match_images = find_matches(template, image, thresh)
    for i in range(len(match_images)):
      cv2.imwrite('output/' + image_name + '.png', match_images[i])

if __name__ == '__main__':
  main(sys.argv[1:])

