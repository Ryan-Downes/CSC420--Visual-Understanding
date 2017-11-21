from skimage import data, feature, color, filters, img_as_float
from matplotlib import pyplot as plt

def DoG(img):
	
	original_image = img_as_float(img)
	img = color.rgb2gray(original_image)

	k = 1.6

	plt.subplot(1,3,1)
	plt.axis('off')
	plt.imshow(original_image)

	for idx,sigma in enumerate([6.0]):
		s1 = filters.gaussian_filter(img,k*sigma)
		s2 = filters.gaussian_filter(img,sigma)

		# multiply by sigma to get scale invariance
		dog = s1 - s2
		plt.subplot(1,3,idx+2)
		plt.axis('off')
		print dog.min(),dog.max()
		plt.imshow(dog,cmap='RdBu')

	ax = plt.subplot(1,3,3)
	ax.axis('off')

	blobs_dog = [(x[0],x[1],x[2]) for x in feature.blob_dog(img, min_sigma=3, max_sigma=6,threshold=0.35,overlap=0)]
	# skimage has a bug in my version where only maxima were returned by the above
	blobs_dog += [(x[0],x[1],x[2]) for x in feature.blob_dog(-img, min_sigma=3, max_sigma=6,threshold=0.35,overlap=0)]

	#remove duplicates
	blobs_dog = set(blobs_dog)

	img_blobs = color.gray2rgb(img)
	for blob in blobs_dog:
		y, x, r = blob
		c = plt.Circle((x, y), r, color='red', linewidth=1, fill=False)
		ax.add_patch(c)
	plt.imshow(img_blobs)

	plt.show()