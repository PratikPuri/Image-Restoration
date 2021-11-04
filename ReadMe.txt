--> To run the code and view the results of image restoration run the file "Image_Restoration_Code.py" by double clicking or opening it in IDLE and running it.

--> When the code runs, it takes "Messi.jpg", present in the folder as the input image. 

--> Images used for training and thus calculation of Sxx are stored in the folder "Training_Images".

--> The output of the code results in 5 different images consisting of:

	a) image_Original : This is the black and white version of the input image given for corruption and then correction.
	b) Blur_image : This is the blurred version of the input image or image_Original.
	c) image_Noise : This is the noise that gets added to the Blur_image.
	d) Corrupted_image : This is the corrupted image that is formed when the image_Noise is added to Blur_image.
	e) Corrected_image : This is the final corrected image obtained as a result of the algorithm used.

--> To close all the images press any button on your keyboard.

--> The PSNR and MSE values for the two different cases is dispayed into the console output.

--> To check for different amounts of noise and blur the value of var in noise and sigma in kernel can be manipulated respectively.
