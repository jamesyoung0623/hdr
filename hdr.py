import cv2
import numpy as np
import random
from scipy import linalg
import math

def triangleWeights(z):
  w = 0.0
  if z < 128:
    w = z + 1.0
  else: 
    w = 256 - z
  return w

def downsample(src):
  dst = cv2.resize(src, (src.shape[0], src.shape[1]), interpolation = cv2.INTER_AREA)
  return dst

def buildPyr(img, maxlevel):
  pyr = []
  pyr.append(img)
  for level in range(maxlevel):
    pyr.append(downsample(pyr[level]))

  return pyr

def getMedian(img):
  hist = cv2.calcHist([img], [0], None, [256], [0, 256])
  median = 0
  sum = 0
  thresh = int(img.shape[0]*img.shape[1]/2)
  while sum < thresh and median < 256:
    sum += int(hist[median])
    median += 1
  
  return median

def computeBitmaps(img):
  median = getMedian(img)

  _, tb = cv2.threshold(img, median, 255, cv2.THRESH_BINARY)
  _, eb = cv2.threshold(abs(img - median), 4, 255, cv2.THRESH_BINARY)
  return tb, eb

def shiftMat(src, shift):
  dst = np.zeros(src.shape)
  width = src.shape[0] - abs(shift[0])
  height = src.shape[1] - abs(shift[1])

  dst[max(shift[0], 0):max(shift[0], 0)+width, max(shift[1], 0):max(shift[1], 0)+height] = src[max(-shift[0], 0):max(-shift[0], 0)+width, max(-shift[1], 0):max(-shift[1], 0)+height]

  return dst

def calculateShift(img0, img1): 
  maxlevel = int(np.log(max(img0.shape[0], img0.shape[1]))/np.log(2.0)) - 1
  maxlevel = min(maxlevel, 5)

  pyr0 = buildPyr(img0, maxlevel)
  pyr1 = buildPyr(img1, maxlevel)  

  shift = (0, 0)
  for i in range(maxlevel+1):
    level = maxlevel-i
    shift *= 2

    tb1, eb1 = computeBitmaps(pyr0[level])
    tb2, eb2 = computeBitmaps(pyr1[level])
    
    min_err = int(pyr0[level].shape[0]*pyr0[level].shape[1])
    new_shift = shift
    for i in range(-1, 2):
      for j in range(-1, 2):
        test_shift = (shift[0]+i, shift[1]+j)

        shifted_tb2 = shiftMat(tb2, test_shift)
        shifted_eb2 = shiftMat(eb2, test_shift)
        
        diff = np.bitwise_xor(tb1.astype(int), shifted_tb2.astype(int))
        diff = np.bitwise_and(diff, eb1.astype(int))
        diff = np.bitwise_and(diff, shifted_eb2.astype(int))
        err = np.count_nonzero(diff)
        
        if err < min_err:
          new_shift = test_shift
          min_err = err
    
    shift = new_shift

  return shift

def AlignMTB(src):
  dst = []
  shifts = []
  pivot = int(len(src)/2)
        
  gray_base = cv2.cvtColor(src[pivot], cv2.COLOR_BGR2GRAY)

  for i in range(len(src)):
    if i == pivot:
      shifts.append((0, 0))
      dst.append(src[pivot])
      continue
    gray = cv2.cvtColor(src[i], cv2.COLOR_BGR2GRAY)
    shift = calculateShift(gray_base, gray)
    shifts.append(shift)
    dst.append(shiftMat(src[i], shift).astype('uint8'))

  return dst


if __name__ == '__main__':
  # Read images and exposure times
  print("Reading images ... ")
  samples = 70
  l = 10.0

  # adjust times and filenames for different scenes
  # place the images and this file in the same directory
  #times = np.array([ 0.001, 0.002, 0.004, 0.01, 0.02, 0.1 ], dtype=np.float32)
  #filenames = ["img_0.001.jpg", "img_0.002.jpg", "img_0.004.jpg", "img_0.01.jpg", "img_0.02.jpg", "img_0.1.jpg"]

  times = np.array([ 0.006, 0.016, 0.033, 0.01, 0.1 ], dtype=np.float32)
  filenames = ["img_0.006.jpg", "img_0.016.jpg", "img_0.033.jpg", "img_0.01.jpg", "img_0.01.jpg"]

  images = []
  R_images = []
  G_images = []
  B_images = []
  R_images_1D = []
  G_images_1D = []
  B_images_1D = []
  
  for filename in filenames:
    im = cv2.imread(filename)
    images.append(im)

  pixel_num = images[0].shape[0]*images[0].shape[1]
  
  print("Aligning images ... ")
  images = AlignMTB(images)
  
  # line 141 to 283 is our algorithm for hdr
  print("Calculating Camera Response Function (CRF) ... ")
  for image in images:
    R_images.append(image[:, :, 0])
    R_images_1D.append(np.reshape(image[:, :, 0], pixel_num))
    G_images.append(image[:, :, 1])
    G_images_1D.append(np.reshape(image[:, :, 1], pixel_num))
    B_images.append(image[:, :, 2])
    B_images_1D.append(np.reshape(image[:, :, 2], pixel_num))

  sample_pixel_idx = sorted(random.sample(range(pixel_num), samples))

  Z_R = []
  Z_G = []
  Z_B = []

  for i in range(len(filenames)):
    sample_R_pixel = [R_images_1D[i][j] for j in sample_pixel_idx]
    Z_R.append(sample_R_pixel)
    sample_G_pixel = [G_images_1D[i][j] for j in sample_pixel_idx]
    Z_G.append(sample_G_pixel)
    sample_B_pixel = [B_images_1D[i][j] for j in sample_pixel_idx]
    Z_B.append(sample_B_pixel)

  Z_R = np.array(Z_R).T 
  Z_G = np.array(Z_G).T 
  Z_B = np.array(Z_B).T 

  B = np.log(times)
  w = triangleWeights
  n = 256

  print('Calculating R channel')
  A = np.zeros((Z_R.shape[0]*Z_R.shape[1]+n+1, n+Z_R.shape[0]))
  b = np.zeros((A.shape[0], 1))

  k = 0 # Include the data-fitting equations
  for i in range(Z_R.shape[0]):
    for j in range(Z_R.shape[1]):
      wij = w(Z_R[i][j])
      A[k][Z_R[i][j]+1] = wij
      A[k][n+i] = -wij
      b[k][0] = wij * B[j]
      k = k+1
  
  A[k][128] = 1 # Fix the curve by setting its middle value to 0
  k = k+1

  for i in range(n-2): # Include the smoothness equations
    A[k][i] = l*w(i+1)
    A[k][i+1] = -2*l*w(i+1)
    A[k][i+2] = l*w(i+1)
    k = k+1

  x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

  E_R = np.zeros((R_images[0].shape[0], R_images[0].shape[1]))
  for i in range(R_images[0].shape[0]):
    for j in range(R_images[0].shape[1]):
      numerator = 0.0
      denominator = 0.0
      for k in range(B.shape[0]):
        numerator += w(R_images[k][i][j])*(x[R_images[k][i][j]][0]-B[k])
        denominator += w(R_images[k][i][j])
      
      ln_E = numerator/denominator
      E_R[i][j] = np.exp(ln_E)
  
  print('Calculating G channel')
  A = np.zeros((Z_G.shape[0]*Z_G.shape[1]+n+1, n+Z_G.shape[0]))
  b = np.zeros((A.shape[0], 1))

  k = 0 # Include the data-fitting equations
  for i in range(Z_G.shape[0]):
    for j in range(Z_G.shape[1]):
      wij = w(Z_G[i][j])
      A[k][Z_G[i][j]] = wij
      A[k][n+i] = -wij
      b[k][0] = wij * B[j]
      k = k+1
  
  A[k][128] = 1 # Fix the curve by setting its middle value to 0
  k = k+1

  for i in range(n-2): # Include the smoothness equations
    A[k][i] = l*w(i+1)
    A[k][i+1] = -2*l*w(i+1)
    A[k][i+2] = l*w(i+1)
    k = k+1

  x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
 
  E_G = np.zeros((G_images[0].shape[0], G_images[0].shape[1]))
  for i in range(G_images[0].shape[0]):
    for j in range(G_images[0].shape[1]):
      numerator = 0.0
      denominator = 0.0
      for k in range(times.shape[0]):
        numerator += w(G_images[k][i][j])*(x[G_images[k][i][j]][0]-np.log(times[k]))
        denominator += w(G_images[k][i][j])

      ln_E = numerator/denominator
      E_G[i][j] = np.exp(ln_E)

  print('Calculating B channel')
  A = np.zeros((Z_B.shape[0]*Z_B.shape[1]+n+1, n+Z_B.shape[0]))
  b = np.zeros((A.shape[0], 1))

  k = 0 # Include the data-fitting equations
  for i in range(Z_B.shape[0]):
    for j in range(Z_B.shape[1]):
      wij = w(Z_B[i][j])
      A[k][Z_B[i][j]] = wij
      A[k][n+i] = -wij
      b[k][0] = wij * B[j]
      k = k+1
  
  A[k][128] = 1 # Fix the curve by setting its middle value to 0
  k = k+1

  for i in range(n-2): # Include the smoothness equations
    A[k][i] = l*w(i+1)
    A[k][i+1] = -2*l*w(i+1)
    A[k][i+2] = l*w(i+1)
    k = k+1

  x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
 
  E_B = np.zeros((B_images[0].shape[0], B_images[0].shape[1]))
  for i in range(B_images[0].shape[0]):
    for j in range(B_images[0].shape[1]):
      numerator = 0.0
      denominator = 0.0
      for k in range(times.shape[0]):
        numerator += w(B_images[k][i][j])*(x[B_images[k][i][j]][0]-np.log(times[k]))
        denominator += w(B_images[k][i][j])

      ln_E = numerator/denominator
      E_B[i][j] = np.exp(ln_E)

  print("Merging images into one HDR image ... ")
  hdrDebevec = cv2.merge((E_R, E_G, E_B))
  hdrDebevec = hdrDebevec.astype('float32') 
  
  #uncomment this part if u want to try opencv implementation
  """
  # Obtain Camera Response Function (CRF)
  calibrateDebevec = cv2.createCalibrateDebevec()
  responseDebevec = calibrateDebevec.process(images, times)
  
  # Merge images into an HDR linear image
  mergeDebevec = cv2.createMergeDebevec()
  hdrDebevec = mergeDebevec.process(images, times, responseDebevec)
  """
  
  # Save HDR image.
  cv2.imwrite("hdrDebevec.hdr", hdrDebevec)
  print("saved hdrDebevec.hdr ")

  # Tonemap using Drago's method to obtain 24-bit color image
  print("Tonemaping using Drago's method ... ")
  tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
  ldrDrago = tonemapDrago.process(hdrDebevec)
  ldrDrago = 3 * ldrDrago
  cv2.imwrite("ldr-Drago.jpg", ldrDrago * 255)
  print("saved ldr-Drago.jpg")
  
  # Tonemap using Reinhard's method to obtain 24-bit color image
  print("Tonemaping using Reinhard's method ... ")
  tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
  ldrReinhard = tonemapReinhard.process(hdrDebevec)
  cv2.imwrite("ldr-Reinhard.jpg", ldrReinhard * 255)
  print("saved ldr-Reinhard.jpg")
  
  # Tonemap using Mantiuk's method to obtain 24-bit color image
  print("Tonemaping using Mantiuk's method ... ")
  tonemapMantiuk = cv2.createTonemapMantiuk(2.2, 0.85, 1.2)
  ldrMantiuk = tonemapMantiuk.process(hdrDebevec)
  ldrMantiuk = 3 * ldrMantiuk
  cv2.imwrite("ldr-Mantiuk.jpg", ldrMantiuk * 255)
  print("saved ldr-Mantiuk.jpg")

