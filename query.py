import requests

while True:
    print()
    query = input('Your query: ')

    results = requests.get(f'http://127.0.0.1:8789/{query}')
    for result in results.json():
        print()
        print(f'Book: {result["id"]}')
        print(f'Certainty: {result["certainty"]}')
        print(f'Location: {result["start"]}-{result["end"]}')
        print(f'Text: {result["text"]})')
