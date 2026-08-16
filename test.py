import requests
fp = r"C:\Users\Ankit Kumar\Downloads\Chapter_7_Bio.pdf"
r = requests.post("http://localhost:8000/ingest/file",
    files={"file": open(fp,"rb")},
    data={"user_id":"Ankit","topic":"Chapter 7 Bio"})
print(r.status_code)
print(r.text)