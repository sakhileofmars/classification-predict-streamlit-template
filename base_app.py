"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
from turtle import width
import streamlit as st
import joblib,os
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
from PIL import Image

# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")








# Set the page width
st.set_page_config(layout="wide",page_title="GreenMind Analytics",
    page_icon="favicon.ico",)




selected2 = option_menu(None, ["Home", "About us", "Predict", 'Analytics','Contact'], 
    icons=['house', 'people-fill', "diagram-3-fill",'graph-up', 'telephone-inbound-fill'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
	    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "grey", "font-size": "20px"}, 
        "nav-link": {"font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "#62d992"},
        "nav-link-selected": {"background-color": "#62d992"},
    })

if selected2 == 'Home':
	lottie_coding = 'https://lottie.host/724173b2-e35f-43f0-88d0-6c065fa20a4f/DkuDmC33f5.json'
	# ---- WHAT I DO ----
	with st.container():
		st.write("---")
		left_column, center_column, right_column = st.columns([1, 4.7, 4.5])


		with center_column:
			st.title("GreenMind Analytics")
			st.write("##")
			st.subheader("Harness the Power of Public Opinion: Unveiling Climate Change Sentiments on Social Media")
			st.write(
				"""Imagine an app that listens to the pulse of the public on climate change, analyzing real-time conversations on social media. 
				This powerful tool, powered by advanced sentiment analysis, can categorize tweets into distinct categories
				"""
			)

		with right_column:
				st_lottie(lottie_coding, height=400, key="coding")

if selected2 == 'About us':
	st.write("---")
	st.subheader("About GreenMind Analytics")
	st.write("##")
	st.write("""
Innovating for a Sustainable Future
Green Mind is a pioneering organization dedicated to accelerating the transition to a sustainable future. We believe that through innovative technology, data-driven insights, and collaborative action, we can empower individuals, communities, and businesses to make informed decisions that benefit both the planet and its people.



Green Mind was born from the belief that addressing climate change and environmental challenges requires more than just awareness. We need actionable insights, informed decision-making, and widespread adoption of sustainable practices. With this vision in mind, we embarked on a mission to develop cutting-edge solutions that enable a greener tomorrow.""")


	with st.container():
		lottie_codings = 'https://lottie.host/e125c274-dacb-4fca-93bb-2a22d86d1b2c/Vj4CeN8Dwa.json'
		st.write("---")
		left_column, center_column, right_column = st.columns([1, 4.7, 4.5])


		with center_column:
			st.subheader("Our Mission")
			st.write(
				"""
				Provide data-driven insights: We leverage cutting-edge technology to gather and analyze vast amounts of public data, providing actionable insights that guide policy decisions and inform sustainable solutions.
				Promote eco-conscious choices: We empower individuals and businesses to make informed choices that contribute to a healthier planet by offering readily accessible resources and promoting sustainable practices.
				Drive widespread adoption: We foster collaboration between individuals, communities, and organizations, working together to create a more sustainable future for all.
								"""
			)

		with right_column:
				st_lottie(lottie_codings, height=400, key="coding")

if selected2 == 'Predict':
	st.write("---")
		# The main function where we will build the actual app
	def main():
		"""Tweet Classifier App with Streamlit """

		# Creates a main title and subheader on your page -
		# these are static across all pages
		st.title("Tweet Classifer")
		st.subheader("Climate change tweet classification")

		# Creating sidebar with selection box -
		# you can create multiple pages this way
		options = [
			"Prediction with Logistic Regression", "Prediction with Random Forest",
			"Prediction with Decision Tree", "Prediction with SVC",
			"Prediction with Tuned LR", "Prediction with Tuned RFC",
			"Information"
		]
		selection = st.selectbox("Choose Option", options)


		# Building out the "Information" page
		if selection == "Information":
			st.info("General Information")
			# You can read a markdown file from supporting resources folder
			st.markdown("Some information here")

			st.subheader("Raw Twitter data and label")
			if st.checkbox('Show raw data'): # data is hidden if box is unchecked
				st.write(raw[['sentiment', 'message']]) # will write the df to the page

		# Building out the predication page
		if selection == "Prediction with Logistic Regression":
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/logreg_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				st.success("Text Categorized as: {}".format(prediction))

		# Building out the predication page
		if selection == "Prediction with Tuned LR":
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/LogReg_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				st.success("Text Categorized as: {}".format(prediction))

		# Building out the predication page
		if selection == "Prediction with Random Forest":
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/RandomForestClassifier_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				st.success("Text Categorized as: {}".format(prediction))

		# Building out the predication page
		if selection == "Prediction with Tuned RFC":
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/randforclf_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				st.success("Text Categorized as: {}".format(prediction))

		# Building out the predication page
		if selection == "Prediction with Decision Tree":
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/DecisionTree_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				st.success("Text Categorized as: {}".format(prediction))

		# Building out the predication page
		if selection == "Prediction with SVC":
			# Creating a text box for user input
			tweet_text = st.text_area("Enter Text","Type Here")

			if st.button("Classify"):
				# Transforming user input with vectorizer
				vect_text = tweet_cv.transform([tweet_text]).toarray()
				# Load your .pkl file with the model of your choice + make predictions
				# Try loading in multiple models to give the user a choice
				predictor = joblib.load(open(os.path.join("resources/SVC_model.pkl"),"rb"))
				prediction = predictor.predict(vect_text)

				# When model has successfully run, will print prediction
				# You can use a dictionary or similar structure to make this output
				# more human interpretable.
				st.success("Text Categorized as: {}".format(prediction))

	# Required to let Streamlit instantiate our web app.  
	if __name__ == '__main__':
		main()

if selected2 == 'Contact':

	st.write("---")

	contact_form = """
	<h4>For more information please contact us...</h4>
	<form action="https://formsubmit.co/your@email.com" method="POST">
     <input type="text" name="name", placeholder="Your name" required>
     <input type="email" name="email", placeholder="Your email" required>
	 <textarea name="message" placeholder="Your message here"></textarea>
     <button type="submit">Send</button>
	</form>

	"""

	st.markdown(contact_form, unsafe_allow_html=True)

	#use local css file
	def local_css(file_name):
			   with open(file_name) as f:
				   st.markdown(f"<style>{f.read()}</styel>", unsafe_allow_html=True)
	
	local_css('style/style.css')













