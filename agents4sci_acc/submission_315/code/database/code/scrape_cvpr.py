import requests
from bs4 import BeautifulSoup
import json
import time
import random
import os
import argparse


def scrape_cvpr_papers(base_url="https://openaccess.thecvf.com", checkpoint_interval=50,year='2025', conference='CVPR', output_dir='./data_original'):
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"{output_dir}/{conference}{year}_papers.json"
    main_page_url = f"{base_url}/{conference}{year}?day=all"
    all_paper_links = []
    papers_data = []

    print(f"Fetching main page: {main_page_url}")
    try:
        response = requests.get(main_page_url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching main page: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')

    # First, collect all paper links
    dt_tags = soup.find_all('dt')
    for dt_tag in dt_tags:
        a_tag = dt_tag.find('a')
        if a_tag and 'href' in a_tag.attrs:
            relative_link = a_tag['href']
            if relative_link.startswith(f'/content/{conference}{year}/html/'):
                all_paper_links.append(f"{base_url}{relative_link}")

    print(f"Found {len(all_paper_links)} paper links. Starting to scrape individual papers with checkpointing...")

    # Load existing data if the output file already exists (for resuming)
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                papers_data = json.load(f)
            print(f"Resuming from {len(papers_data)} previously scraped papers.")
        except json.JSONDecodeError:
            print("Existing output file is corrupted or empty. Starting fresh.")
            papers_data = []

    scraped_urls = {paper["page_url"] for paper in papers_data}

    for i, paper_url in enumerate(all_paper_links):
        if paper_url in scraped_urls:
            print(f"  Skipping already scraped paper: {paper_url}")
            continue

        print(f"  Scraping paper {i+1}/{len(all_paper_links)}: {paper_url}")
        
        try:
            paper_response = requests.get(paper_url)
            paper_response.raise_for_status()
            paper_soup = BeautifulSoup(paper_response.text, 'html.parser')

            title = paper_soup.find('div', id='papertitle').text.strip() if paper_soup.find('div', id='papertitle') else 'N/A'
            authors = paper_soup.find('div', id='authors').text.strip() if paper_soup.find('div', id='authors') else 'N/A'
            abstract = paper_soup.find('div', id='abstract').text.strip() if paper_soup.find('div', id='abstract') else 'N/A'
            
            pdf_link_tag = paper_soup.find('a', href=lambda href: href and href.endswith('.pdf'))
            pdf_link = f"{base_url}{pdf_link_tag['href']}" if pdf_link_tag else 'N/A'

            papers_data.append({
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "pdf_link": pdf_link,
                "page_url": paper_url
            })
            scraped_urls.add(paper_url)

            # Checkpoint saving
            if len(papers_data) % checkpoint_interval == 0:
                print(f"  Checkpoint: Saving {len(papers_data)} papers to {output_file}")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(papers_data, f, ensure_ascii=False, indent=4)

            time.sleep(random.random()) # Random delay between 0 and 1 second

        except requests.exceptions.RequestException as e:
            print(f"    Error fetching paper page {paper_url}: {e}")
        except Exception as e:
            print(f"    Error parsing paper page {paper_url}: {e}")

    # Final save after all papers are processed
    if papers_data:
        print(f"Finished scraping. Final save of {len(papers_data)} papers to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(papers_data, f, ensure_ascii=False, indent=4)
    else:
        print("No papers were scraped.")

    return papers_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape paper data from CVF Open Access.')
    parser.add_argument('--year', type=int, required=True, help='Conference year, e.g., 2024')
    parser.add_argument('--conference', type=str, required=True, help='Conference name, e.g., CVPR')
    parser.add_argument('--outdir', type=str, required=True, help='Path to save the json file')
    args = parser.parse_args()
    scrape_cvpr_papers(year=str(args.year), conference=args.conference, output_dir=args.outdir)
