'''
Python Webscraping Beautiful Soup

Webscraping sandbox with Beautiful Soup and requests modules.
'''

# Module Imports
from bs4 import BeautifulSoup
import requests

# Constants
URL = 'https://www.monster.co.uk/jobs/search/?q=Python-Developer&where=Liverpool__2C-NW'

# Retrieve page
page = requests.get(URL)

# Start parsing
soup = BeautifulSoup(page.content, 'html.parser')

# Find Elements of interest
results = soup.find(id = 'ResultsContainer')

# Find job elements
job_elems = results.find_all('section', class_ = 'card-content')

# Print data for each element
for job_elem in job_elems:
    # Each job_elem is a new BeautifulSoup object.
    # You can use the same methods on it as you did before.
    try:
        title_elem = job_elem.find('h2', class_ = 'title')
        company_elem = job_elem.find('div', class_ = 'company')
        location_elem = job_elem.find('div', class_ = 'location')
        if None in (title_elem, company_elem, location_elem):
            continue
        print(title_elem.text.strip())
        print(company_elem.text.strip())
        print(location_elem.text.strip())
        print() # Add a space

    except:
        pass

# Find results for python developers
engineer_jobs = results.find_all('h2', 
                                string = lambda text: 'engineer' in text.lower())
print(len(engineer_jobs))

# Return links for jobs of interest
for job in engineer_jobs:
    link = job.find('a')['href']
    print(job.text.strip())
    print(f"Apply here: {link}\n")