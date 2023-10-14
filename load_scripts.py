import os
import json
import requests
from bs4 import BeautifulSoup

def extract_data(entry):
    character = entry.find('b').text.strip(':')
    line = entry.text.strip().replace(f'{character}:', '').strip()
    line = line.replace("\n", "")
    return {'character': character, 'line': line}

def scrape_script(url, output_file):
    # Send a request to the website and get the HTML content
    response = requests.get(url)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Example: Extract all the text inside <p> tags
        target_elements = soup.find_all(lambda tag: tag.name == 'p' and tag.b and ':' in tag.b.text)
        
        parsed_data = []
        for idx, item in enumerate(target_elements, start=1):
            data = extract_data(item)
            data['id'] = idx
            parsed_data.append(data)

        # Save the data to a JSON file
        with open(output_file, 'w', encoding='utf-8') as json_file:
            json.dump(parsed_data, json_file, indent=4, ensure_ascii=False)

    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")

def scrape_links(url, season):
    # Send a request to the website and get the HTML content
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        season_p = soup.find('p', text='Season '+season+':')
        links = season_p.find_next_sibling().find('ul').find_all('a', href=True)
        href_links = [link['href'] for link in links]
        return href_links
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")

if __name__ == "__main__":
    # HYPERPARAMS
    base_url = 'https://www.livesinabox.com/friends'
    season = "9"
    output_folder = "../Dataset/S09/Script"
    input_folder = "../Dataset/S09/Audio"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # URL of the webpage to scrape
    links = scrape_links(base_url+"/scripts.shtml", season)

    season = "0" + season if len(season) == 1 else season
    episodes_name = os.listdir(input_folder)
    episodes_name = [e.split(".")[0] for e in episodes_name]
    for i,link in enumerate(links) :
        if i < len(episodes_name):
            script_link = base_url+"/"+link
            
            output_file = output_folder+"/"+episodes_name[i]+".json"
            scrape_script(script_link,output_file)