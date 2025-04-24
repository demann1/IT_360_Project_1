# IT_360_Project_1
This is the ML phishing email detection bot code for our IT360 Final project. 

To run this, open an integrated terminal in your IDE or simply download the file and then run it from CMD.
"python main.py" and "python main.py --local" are the only two acceptable running scripts. 

This bot has two modes of running, normal, and local. Running without the "--local" tag will run off a 
model database with 200 labeled emails. The bot works by predicting if the email is a phishing email or
not and outputs a simple output showing how many were predicted correctly, this often triggers 100% accuracy
however droping the sample size down to a more reasonable 30-80 seems to be best for real world resemblence. 
Running this in local mode uses the dataset as a sample to train with, and then calls the Google API to draw 
emails from your gmail inbox. So far there is only 1 user tied to this API so this would take some altering to 
run from your computer. I would recommend changing the API call to your own Google API and allowing yourself
access that way. 

This code is not perfect or proofed for security, use at your own risk.