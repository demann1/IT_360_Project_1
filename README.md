# IT_360_Project_1
The goal of this project is to create a ML model that will detect phishing emails as well as spam emails.

To do this, we downloaded a .csv file with 200 sample emails labeled phishing 1 for phishing and 0 for safe. For the spam we will do a similar process flow, implement a labeled .csv list from an online source to train our ML model on and verify its detection rates. 

A successful outcome is the ML bot determining with 80% accuracy a fishing or spam email

To run this, running in --local mode calls the google API to the authorized users inbox to run phishing email scans on personal inboxes. This works by calling the API with the 'credentials.json' in the config folder. To run this, create your own API hook and add the credentials file in. 