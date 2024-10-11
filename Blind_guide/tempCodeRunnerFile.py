import requests

# Define the API endpoint
url = "http://[2402:4000:11c0:def5:6414:e448:de1:25fa]:5000/transcribe"

# The text to send
text = "Hi my name is sri lanaka"

# Construct the query parameters
params = {
    'text': text
}

# Send the GET request to the API
response = requests.get(url, params=params)

# Print the response from the server
if response.status_code == 200:
    print("Response:", response.json())
else:
    print(f"Error: {response.status_code}, {response.text}")
