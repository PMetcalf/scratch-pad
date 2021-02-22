'''
Python API Library

Best practices for building an API Library to consume or provision an API.
'''

# Module imports
import requests

# Constants
base_url = 'https://todolist.example.com/tasks/'

# --- Using API - Getting Data (Get Task) ---

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
# (Could also be a module)

class todo:

    def _url(self, path):
        return 'https//todo.example.com' + path

    def get_tasks(self):
        return requests.get(_url('/tasks/'))
    
    def describe_task(self, task_id):
        return requests.get(_url('/tasks/{:d}/'.format(task_id)))

    def add_task(self, summary, description = ""):
        return requests.post(_url('/tasks/'), json = {
            'summary': summary,
            'Description': description,
        })

    def task_done(self, task_id):
        return requests.delete(_url('/tasks/{:d}'.format(task_id)))

    def update_task(self, task_id, summary, description):
        url = _url('tasks/{:d}'.format(task_id))
        return requests.put(url, json = {
            'summary': summary,
            'description': description,
        })