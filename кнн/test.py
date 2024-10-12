import requests

proxies = {
    'http': 'http://your-proxy-address:port',
    'https': 'https://your-proxy-address:port'
}

response = requests.get('https://chat.openai.com/chat', proxies=proxies)
print(response.text)