import requests

limit = 1

while True:
    print()
    query = input('Your query: ')

    results1 = requests.get(f'http://0.0.0.0:8791/v2/?l={limit}&q={query}')
    results2 = requests.get(f'http://0.0.0.0:8791/v2/l={limit}&q={query}')
    results3 = requests.get(f'http://0.0.0.0:8790/v2/l={limit}&q={query}')
#    results = requests.get(f'http://0.0.0.0:8791/{query}')

# print(results)
    for result in results1.json():
        print()
        print("RESULT 1")
        print()
        print(f'Book: {result["id"]}')
        print(f'Certainty: {result["certainty"]}')
        print(f'Location: {result["start"]}-{result["end"]}')
        print(f'Text: {result["text"]})')
    for result in results2.json():
        print()
        print("RESULT 2")
        print()
        print(f'Book: {result["id"]}')
        print(f'Certainty: {result["certainty"]}')
        print(f'Location: {result["start"]}-{result["end"]}')
        print(f'Text: {result["text"]})')
    for result in results3.json():
        print()
        print("RESULT 3")
        print()
        print(f'Book: {result["id"]}')
        print(f'Certainty: {result["certainty"]}')
        print(f'Location: {result["start"]}-{result["end"]}')
        print(f'Text: {result["text"]})')
