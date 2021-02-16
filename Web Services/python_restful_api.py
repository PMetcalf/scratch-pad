'''
Python RESTful API

Working with a simple RESTful API sandbox with request module.
'''

# Module Imports
from datetime import datetime
import json
import requests

astros_url = "http://api.open-notify.org/astros.json"
passover_url = "http://api.open-notify.org/iss-pass.json"

def json_print(obj):
    # Create a formatted string of a json object
    text = json.dumps(obj, sort_keys = True, indent = 4)
    print(text)

#--- Simple Response - Astronauts ---

# Get astros response
response = requests.get(astros_url)

# Print astros response
print(response.status_code)
json_print(response.json())

#--- Paramater Response - Passover ---

# Set latitude and longitude
lat_long_params = {
    "lat": 53.3,
    "lon": 3.08
}

# Get passover response
passover_response = requests.get(passover_url, params = lat_long_params)

# Print passover response
print(passover_response.status_code)
json_print(passover_response.json())

# Extract passover times
pass_times = passover_response.json()['response']
json_print(pass_times)

# Extract rise times from passover data
risetimes = []

for data in pass_times:
    time = data['risetime']
    risetimes.append(time)

# Prints times as epochs
print(risetimes)

# Convert epochs time
times = []

for rt in risetimes:
    time = datetime.fromtimestamp(rt)
    times.append(time)
    print(time)