my_url = 'https://scikit-learn.org/stable/modules/clustering.html#clustering'
#Get html content
import requests
html_page = requests.get(my_url)
#Html parsering
from bs4 import BeautifulSoup
soup = BeautifulSoup(html_page.content, 'html.parser')
#Get the table content
containers = soup.findAll('div', class_='section', id='overview-of-clustering-methods')
container = containers[0]
#Get the title of the table
title = container.p.text
#Open file for writing
import csv
fh = open('Fetch_result.csv', 'w')
fh_writer = csv.writer(fh, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#Get the content of the table
table = container.findAll('tr')
for i in table:
    content = i.text.strip().split('\n')
    #Write to a file
    fh_writer.writerow(content)
fh.close()
print('Information successfully saved to "Fetch_result.csv" file')


