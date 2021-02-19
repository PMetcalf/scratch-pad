'''
Python API Library

Best practices for building an API Library to consume or provision an API.
'''

# Module imports
import requests

# Constants
base_url = 'https://todolist.example.com/tasks/'

# --- Getting Data (Get Task) ---

# Get task
resp = requests.get(base_url)

# Handle erroneous responses
if resp.status_code != 200:
    raise ApiError('GET /tasks/ {}'.format(resp.status_code))

# Create task object from response
for todo_item in resp.json():
    print('{} {}'.format(todo_item['id'], todo_item['summary']))

# --- Posting Data (Post Task) ---

# Define task
task = {"summary": "Take out trash", "description": ""}

# Post task
resp = requests.post(base_url, json = task)

# Alternate post routine (XML, YAML)
#respo = requests.post(base_url,
#                      data = json.dumps(task),
#                      headers = {'Content-Type':'application/json'},)  # Adds key as new header field

# Handle erroneous response
if resp.status_code != 201:
    raise ApiError('POST /tasks/ {}'.format(resp.status_code))

print('Created task, ID: {}'.format(resp.json()["id"]))

# --- API Library (For Tasks) ---

class todo:

