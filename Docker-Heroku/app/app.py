import streamlit as st
import cv2
from PIL import Image, ImageEnhance, ImageFont, ImageDraw
import numpy as np
import os
import pandas as pd
from keras.models import load_model
from joblib import load
import itertools
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage as ndi
from mtcnn import MTCNN



@st.cache(show_spinner=False)
def load_image(img):
	return Image.open(img)

def image_enhancers(image_file):
	images = []
	enhance_type = st.sidebar.radio("Enhance Type",["Original","Gray-Scale","Contrast","Brightness","Blurring"])
	if enhance_type == 'Gray-Scale':
		st.text("Gray-Scale Image")
		for img in image_file:
			image = Image.open(img)
			new_img = np.array(image.convert('RGB'))
			img = cv2.cvtColor(new_img,1)
			output = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			image_center(output)
			images.append(output)
	elif enhance_type == 'Contrast':
		c_rate = st.sidebar.slider("Contrast",0.2,10.0)
		st.text("Contrast Image with {} contrast".format(c_rate))

		for img in image_file:
			image = Image.open(img)
			enhancer = ImageEnhance.Contrast(image)
			output = enhancer.enhance(c_rate)
			image_center(output)
			images.append(output)
	elif enhance_type == 'Brightness':
		c_rate = st.sidebar.slider("Brightness",0.2,10.0)
		st.text("Bright Image with {} brightness".format(c_rate))
		for img in image_file:
			image = Image.open(img)
			enhancer = ImageEnhance.Brightness(image)
			output = enhancer.enhance(c_rate)
			image_center(output)
			images.append(output)
	elif enhance_type == 'Blurring':
		blur_rate = st.sidebar.slider("Brightness",0.2,10.0)
		st.text("Blurred Image with {} blurr rate".format(blur_rate))
		for img in image_file:
			image = Image.open(img)
			new_img = np.array(image.convert('RGB'))
			img = cv2.cvtColor(new_img,1)
			output = cv2.GaussianBlur(img,(11,11),blur_rate)
			image_center(output)
			images.append(output)

	elif enhance_type == 'Original':

		st.text("Original Images")
		for img in image_file:
			image = Image.open(img)
			image_center(image)
			images.append(image)
	else:
		st.text("Original Images")
		for img in image_file:
			image = Image.open(img)
			image_center(image)
			images.append(image)
	return images

def image_center(image, col_l= "", col_r=""):
	col1, col2, col3 = st.columns([1,3,15])
	with col1:
		st.write(col_l)

	with col2:
		st.image(image, width=400)

	with col3:
		st.markdown(col_r, unsafe_allow_html=True)

@st.cache(show_spinner=False)
def process_img2(img, img_size):
    face = MTCNN().detect_faces(img)       # Assuming only one face is given
    if len(face) != 0:
        x1, y1, width, height = face[0]['box']
        x2, y2 = x1 + width, y1 + height
        new_img = img[y1:y2, x1:x2]
    else:
        new_img = img

    new_img = cv2.resize(new_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    new_img = np.array(new_img) / 255.0
    return  new_img


@st.cache(show_spinner=False)
def predicting_images(images, model, class_type):

	#classes = get_image_classes(3)
	ID_GENDER_MAP = {0: 'male', 1: 'female'}
	ID_RACE_MAP = {0: 'white', 1: 'black', 2: 'asian', 3: 'indian', 4: 'others'}
	ID_AGEGROUP_MAP = {0 : 'baby/kid', 1 : 'teenager',\
                       2 : 'young_adult', 3 : 'adult', 4 : 'senior'}
	#font = ImageFont.truetype("Roboto-Regular.ttf", 50)
	text_to_show = []
	all_preds = []
	for image in images:

		#Resize imgae to meet input shape
		img = np.array(image)

		if len(img.shape) < 3:
			text_to_show.append('Does not work with GrayScale images')
			continue

		img_size = 128 #200
		pred_data = process_img2(img, img_size)

		# get model predictions with probabilities
		pred1, pred2, pred3 = model.predict(pred_data.reshape(1, img_size, img_size, 3))
		pred1, pred2, pred3 = pred1.argmax(axis=-1)[0], pred2.argmax(axis=-1)[0], pred3.argmax(axis=-1)[0]
		all_preds.append([pred1, pred2, pred3])


		if class_type == 'Gender':
			text_to_show.append(ID_GENDER_MAP[pred2])
		elif class_type == 'Race':
			text_to_show.append(ID_RACE_MAP[pred3])
		elif class_type == 'Age group':
			text_to_show.append(ID_AGEGROUP_MAP[pred1])
		elif class_type == 'All':
			text_to_show.append(ID_GENDER_MAP[pred2] + ' ' + \
							    ID_RACE_MAP[pred3] + ' '  + \
							    ID_AGEGROUP_MAP[pred1])
	return text_to_show, all_preds

@st.cache(show_spinner=False)
def process_img(img, img_size):

    # Sharpen image to avoid null va;ues for very blurry images
    kernel = np.array([[0, -1, 0],[-1, 5,-1],[0, -1, 0]])
    img2 = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

    img2 = cv2.resize(img2, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img2 = cv2.Canny(img2, 100, 200, 3, L2gradient=True)
    img2 = (img2 - img2.mean()) / img2.std()
    img2 = window_mean_std(img2)



    img3 = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    # Gabor feature
    ksize = 15
    lamda = np.pi/4
    theta = np.pi/4
    sigma = 1.0
    phi = 1
    gamma = 0.5

    kernel = cv2.getGaborKernel((ksize, ksize),sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(img3, cv2.CV_8UC3, kernel)
    img3  = power(filtered_img, kernel)
    img3 = (img3 - img3.mean()) / img3.std()
    img3 = window_mean_std(img3)
    return  list(np.array([img2, img3]).flatten())


@st.cache(show_spinner=False)
def window_mean_std(new_img):
    # Divide the (img_size, img_size) pixels into (10, 10) pixels each
    # Obtain the mean and standard deviation of each (10, 10) pixel
    # This results in a (400, 2) array to be used
    window = 10
    new_win = int(new_img.shape[0]/window)
    arr = sliding_window_view(new_img, window_shape = (window, window))[::window, ::window].reshape((new_win*new_win, window*window))
    edges_window_mean = np.mean(arr, axis=1)
    edges_window_std = np.std(arr, axis=1)

    return np.array([edges_window_mean, edges_window_std]).T

@st.cache(show_spinner=False)
def power(image, kernel):
    # Normalize images for better comparison.
    image = (image - image.mean()) / image.std()
    return np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                   ndi.convolve(image, np.imag(kernel), mode='wrap')**2)

@st.cache(show_spinner=False)
def predict_age(images, age_model, pred_vals):
	model = age_model[0]
	best_cols = age_model[1]
	dummy_cols = best_cols[-12:]
	k = 0
	img_size = 128 # 200
	preds = []
	for image in images:
		age_group = [1 if pred_vals[k][0] == i else 0 for i in range(5)]
		gender = [1 if pred_vals[k][1] == i else 0 for i in range(2)]
		race = [1 if pred_vals[k][2] == i else 0 for i in range(5)]
		temp = process_img(np.array(image), img_size)
		num = len(temp)
		vals = np.array(temp  + age_group + gender + race).reshape((1, -1))
		columns = ['X' + str(i) for i in range(num)] + dummy_cols
		age_df = pd.DataFrame(vals, columns=columns)
		age  = model.predict(age_df[best_cols])[0]
		preds.append(age)
		k += 1
	return preds


def image_with_labels(images, text_list, ages):
	i = 0
	for image in images:
		image = np.array(image)
		font = cv2.FONT_HERSHEY_COMPLEX
		t = "#### <span style='text-align: center; color: blue;'>{}<br> Estimated age: {}</span>".format(text_list[i], str(int(ages[i])))
		image_center(image, col_r=t)
		i += 1


@st.cache(show_spinner=False)
def detect_eyes(our_image):
	img = np.array(our_image)
	eyes = eye_cascade.detectMultiScale(img, scaleFactor = 1.2, minNeighbors = 4)
	for (x,y,w,h) in eyes:
	        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),5)
	return img, eyes

@st.cache(show_spinner=False)
def detect_smiles(our_image):
	img = np.array(our_image)
	smiles = smile_cascade.detectMultiScale(img, 1.1, 4)
	# Draw rectangle around the Smiles
	for (x, y, w, h) in smiles:
	    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
	return img, smiles

@st.cache(show_spinner=False)
def detect_noses(our_image):
	img = np.array(our_image)
	noses = nose_cascade.detectMultiScale(img, 1.1, 4)

	for (x, y, w, h) in noses:
	    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
	return img, noses

@st.cache(show_spinner=False)
def detect_mouths(our_image):
	img = np.array(our_image)
	mouths = mouth_cascade.detectMultiScale(img, 1.1, 4)
	# Draw rectangle around the Smiles
	for (x, y, w, h) in mouths:
	    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
	return img, mouths

WORKDIR = '/app/'
eye_cascade = cv2.CascadeClassifier(WORKDIR + 'haarcascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(WORKDIR +'haarcascades/haarcascade_smile.xml')
mouth_cascade  = cv2.CascadeClassifier(WORKDIR +'haarcascades/haarcascade_mcs_mouth.xml')
nose_cascade  = cv2.CascadeClassifier(WORKDIR + 'haarcascades/haarcascade_mcs.nose.xml')




def main():

	"""Face Detection App"""
	st.title("Image Detection and Prediction App")
	st.text("Built with Streamlit, OpenCV and TensorFlow")

	activities = ["Detection", "Prediction", "About"]
	choice = st.sidebar.selectbox("Select Activity",activities)
	model = load_model(WORKDIR + 'models/model.h5')
	age_model = load(WORKDIR + 'models/finalized_model.joblib')
	page = None
	if choice != 'About':
		page_names = ['Upload files', 'Use your camera']
		page = st.radio('', page_names)


	image_files = []
	if page == 'Upload files':
		st.markdown(""" <style> .font {
		font-size:35px ; font-family: 'Cooper Black'; color: #ADD8E6;}
		</style> """, unsafe_allow_html=True)
		st.markdown('<p class="font">Upload your photos here...</p>', unsafe_allow_html=True)
		image_files = st.file_uploader("",type=['jpg','png','jpeg'], accept_multiple_files=True)

	if page == 'Use your camera':
		st.markdown(""" <style> .font {
		font-size:35px ; font-family: 'Cooper Black'; color: #ADD8E6;}
		</style> """, unsafe_allow_html=True)
		st.markdown('<p class="font">Take a picture of yourself here...</p>', unsafe_allow_html=True)
		image_file = st.camera_input('Smile at the camera now ...', st.image('Smiley.jpeg', width=50))

		if image_file:
			image_files = [image_file] # Pass as list since the uploader accept multiple files as list

	if choice == 'About':
			st.subheader("About Image Detection and Prediction App")
			st.markdown("Built with Streamlit, OpenCV and Tensorflow by  Likedream (T. A. Biala)")
			st.text("T. A. Biala")

	elif choice == 'Detection':
		if image_files:
			images = image_enhancers(image_files)
		st.subheader("Face Detection")
		# Face Detection
		task = ["Eyes","Noses","Mouths","Smiles"]
		feature_choice = st.sidebar.selectbox("Features to detect",task)
		if st.button("Process"):
			if image_files:
				if feature_choice == 'Smiles':
					for image in images:
						img, length = detect_smiles(image)
						st.image(img, width=400)
						st.success("Found {} {}".format(len(length), feature_choice))
				elif feature_choice == 'Eyes':
					for image in images:
						img, length = detect_eyes(image)
						st.image(img, width=400)
						st.success("Found {} {}".format(len(length), feature_choice))
				elif feature_choice == 'Noses':
					for image in images:
						img, length = detect_noses(image)
						st.image(img, width=400)
						st.success("Found {} {}".format(len(length), feature_choice))
				elif feature_choice == 'Mouths':
					for image in images:
						img, length = detect_mouths(image)
						st.image(img, width=400)
						st.success("Found {} {}".format(len(length), feature_choice))

			else:
				st.write('No file found, try upload an image file to process')

	elif choice == 'Prediction':
		st.text("Original Images")
		images = []
		for img in image_files:
			image = Image.open(img)
			image_center(image)
			images.append(image)
		st.subheader("Image Prediction")
		predictions = ['Gender', 'Race', 'Age group', 'All']
		feature_choice = st.sidebar.selectbox("Features to predict", predictions)
		if st.button("Predict"):
			col1, col2 = st.columns([3,20])
			with col1:
				st.write("")
			with col2:
				st.markdown(" Running Predictions... ")
				#st.markdown('<h6 class="font"> Running Predictions... </h6>', unsafe_allow_html=True)

			if image_files:
				text_list, pred_vals = predicting_images(images, model, feature_choice)
				ages = predict_age(images, age_model, pred_vals)
				image_with_labels(images, text_list, ages)





	#Add a feedback section in the sidebar
	st.sidebar.title(' ') #Used to create some space between the filter widget and the comments section
	st.sidebar.markdown(' ') #Used to create some space between the filter widget and the comments section
	st.sidebar.subheader('Please help us improve!')
	with st.sidebar.form(key='columns_in_form',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
	    rating=st.slider("Please rate the app", min_value=0, max_value=5, value=3,help='Drag the slider to rate the app. This is a 0-5 rating scale where 5 is the highest rating')
	    text=st.text_input(label='Please leave your feedback here:')
	    submitted = st.form_submit_button('Submit')
	    if submitted:
	      st.write('Thanks for your feedback!')
	      st.markdown('Your Rating:')
	      st.markdown(rating)
	      st.markdown('Your Feedback:')
	      st.markdown(text)


if __name__ == '__main__':
		main()
