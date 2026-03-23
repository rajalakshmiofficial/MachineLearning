import requests

url = "http://127.0.0.1:8000/predict"   # update port if different

payload = {
    "age": 35,
    "sex": "female",
    "bmi": 27.5,
    "children": 2,
    "smoker": "no",
    "region": "southwest"
}

try:
    response = requests.post(url, json=payload)
    print("Status Code:", response.status_code)
    print("Response:", response.json())

except Exception as e:
    print("Error occurred:", str(e))
