'''
Python API Integration

Best practices for integrating an API into a workflow.
'''

# Module imports
import requests

# Constants
base_url = 'https://todolist.example.com/tasks/'

# --- Getting data ---

resp = requests.get(base_url)

# Handle erroneous responses
if resp.status_code != 200:
    raise ApiError('GET /tasks/ {}'.format(resp.status_code))

for todo_item in resp.json():
    print('{} {}'.format(todo_item['id'], todo_item['summary']))

