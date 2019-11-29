# import the necessary packages
import numpy as np
import cv2

class CropPreprocessor(object):
	def __init__(self, width, height, horiz=True, inter=cv2.INTER_AREA):
		# store the target image width, height, whether or not
		# horizontal flips should be included, along with the
		# interpolation method used when resizing
		self.width = width
		self.height = height
		self.horiz = horiz
		self.inter = inter

	def __call__(self, sample):
		# initialize the list of crops
		image,mask = sample['O'],sample['M']
		crops = []
		mask_crops = []

		# grab the width and height of the image then use these
		# dimensions to define the corners of the image based
		(h, w) = image.shape[:2]
		coords = [
			[0, 0, self.width, self.height],
			[w - self.width, 0, w, self.height],
			[w - self.width, h - self.height, w, h],
			[0, h - self.height, self.width, h]]

		# compute the center crop of the image as well
		dW = int(0.5 * (w - self.width))
		dH = int(0.5 * (h - self.height))
		coords.append([dW, dH, w - dW, h - dH])

		# loop over the coordinates, extract each of the crops,
		# and resize each of them to a fixed size
		for (startX, startY, endX, endY) in coords:
			crop = image[startY:endY, startX:endX]
			crop = cv2.resize(crop, (self.width, self.height),
				interpolation=self.inter)
			crops.append(crop)

			mask_crop = mask[startY:endY, startX:endX]
			mask_crop = cv2.resize(mask_crop, (self.width, self.height),
				interpolation=self.inter)
			mask_crops.append(mask_crop)

		# check to see if the horizontal flips should be taken
		if self.horiz:
			# compute the horizontal mirror flips for each crop
			mirrors = [cv2.flip(c, 1) for c in crops]
			crops.extend(mirrors)

			mask_mirrors = [cv2.flip(c, 1) for c in mask_crops]
			mask_crops.extend(mask_mirrors)

		# return the set of crops
		return {'O':crops,'M':mask_crops}