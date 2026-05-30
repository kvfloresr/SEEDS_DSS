import requests
url = "http://127.0.0.1:5000/predict"
files = {'image': open('test_images/100broken.bmp','rb')}
r = requests.post(url, files=files)
print(r.json())
