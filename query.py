import requests

while True:
    print()
    query = input('Your query: ')

    results = requests.get(f'http://epsilon-it.utu.fi/ecco-api/{query}')
    for result in results.json():
        print()
        print(f'Book: {result["id"]}')
        print(f'Certainty: {result["certainty"]}')
        print(f'Location: {result["start"]}-{result["end"]}')
        print(f'Text: {result["text"]})')
