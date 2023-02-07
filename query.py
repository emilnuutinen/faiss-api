import requests

limit = 20

while True:
    print()
    query = input('Your query: ')

    results = requests.get(f'http://0.0.0.0:8791/v2/l={limit}&q={query}')
    
    for result in results.json():
        print()
        print(f'Book: {result["id"]}')
        print(f'Certainty: {result["certainty"]}')
        print(f'Location: {result["start"]}-{result["end"]}')
        print(f'Text: {result["text"]})')
