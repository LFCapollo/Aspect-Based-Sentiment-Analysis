import requests
from pprint import  pprint

url = 'http://127.0.0.1:5000/process'
data = {"review":"Primantis has changed a lot over the years. They now have more options than ever, you don't order at the kitchen counter anymore."
                 " (Someone will come take your order). This location is the most authentic to the original Primantis."
                 " It's kindof divey & not overly corporate like the ones outside of the city. Quick service, same taste as always."
                 " This is a casual setting where you can grab a quick bite, or bring your friends to hang out for a couple beers, and load up on carbs."}
r = requests.post(url, json=data)

pprint(r.json())
