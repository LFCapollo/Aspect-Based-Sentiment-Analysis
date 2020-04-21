import requests

url = 'http://127.0.0.1:5000/process'
r = requests.post(url, json={"review":"I will most likely get slain to a cross for this, but I am not a huge fan of Primanti's. The fries are outstanding and fresh, as well as the Italian bread they put the sandwiches on (which is distributed from a local bakery), and the pastrami is juicy and delicious."})

print(r.json())