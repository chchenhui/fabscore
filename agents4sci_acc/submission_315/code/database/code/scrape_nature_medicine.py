
import requests
from bs4 import BeautifulSoup
import json
import time
import random
import os
import re
import argparse

def scrape_nature_medicine_papers(
    base_url="https://www.nature.com",
    start_year=2025,
    end_year=2025, # You can change this to scrape multiple years
    output_file="nature_medicine_papers.json",
    links_file="nature_medicine_article_links.json", # New file for storing links
    checkpoint_interval=50,
    start_page=1 # For resuming pagination from a specific page
):
    all_article_links = []
    papers_data = []

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

    # Try to load article links from file first
    if os.path.exists(links_file):
        try:
            with open(links_file, 'r', encoding='utf-8') as f:
                all_article_links = json.load(f)
            print(f"Loaded {len(all_article_links)} article links from {links_file}.")
        except json.JSONDecodeError:
            print(f"Existing links file {links_file} is corrupted or empty. Re-collecting links.")
            all_article_links = []
    
    if not all_article_links: # If links were not loaded or file was corrupted, collect them
        print("--- Collecting all article links across pages ---")
        for year in range(start_year, end_year + 1):
            current_page = start_page
            while True:
                # Construct the URL for the current listing page
                if current_page == 1:
                    listing_url = f"{base_url}/nm/articles?type=article&year={year}"
                else:
                    listing_url = f"{base_url}/nm/articles?searchType=journalSearch&sort=PubDate&type=article&year={year}&page={current_page}"
                
                print(f"Fetching listing page (Year: {year}, Page: {current_page}): {listing_url}")
                try:
                    response = requests.get(listing_url)
                    response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    print(f"Error fetching listing page {listing_url}: {e}")
                    break # Stop if a listing page cannot be fetched

                soup = BeautifulSoup(response.text, 'html.parser')

                # Find article links on the current listing page
                article_link_tags = soup.select('h3.c-card__title a')
                if not article_link_tags:
                    print(f"No more article links found on page {current_page}. Moving to next year or finishing.")
                    break # No more articles on this page, or end of pagination

                for link_tag in article_link_tags:
                    relative_link = link_tag.get('href')
                    if relative_link and relative_link.startswith('/articles/'):
                        full_link = f"{base_url}{relative_link}"
                        if full_link not in all_article_links:
                            all_article_links.append(full_link)
                
                # Find the next page link
                # next_page_link = soup.find('a', class_='c-pagination__link', attrs={'data-page': 'next'})
                # if next_page_link and next_page_link.get('href'):
                current_page += 1
                time.sleep(random.random()) # Random delay between listing pages
                #else:
                #    print(f"No 'Next' page link found for year {year}, page {current_page}.")
                #    break # No more pages
        
        # Save collected links to file
        if all_article_links:
            with open(links_file, 'w', encoding='utf-8') as f:
                json.dump(all_article_links, f, ensure_ascii=False, indent=4)
            print(f"Saved {len(all_article_links)} article links to {links_file}.")
        else:
            print("No article links were collected.")

    print(f"Collected {len(all_article_links)} unique article links. Starting to scrape individual articles...")

    # Iterate through collected links and scrape individual article details
    for i, article_url in enumerate(all_article_links):
        if article_url in scraped_urls:
            print(f"  Skipping already scraped article: {article_url}")
            continue

        print(f"  Scraping article {i+1}/{len(all_article_links)}: {article_url}")
        
        try:
            article_response = requests.get(article_url)
            article_response.raise_for_status()
            article_soup = BeautifulSoup(article_response.text, 'html.parser')

            title_tag = article_soup.find('h1', class_='c-article-title')
            title = title_tag.text.strip() if title_tag else 'N/A'

            # --- MODIFIED: Extract authors from meta tags --- 
            authors_meta = article_soup.find_all('meta', attrs={'name': 'dc.creator'})
            authors_list = [meta['content'] for meta in authors_meta if 'content' in meta.attrs]
            authors = '; '.join(authors_list) if authors_list else 'N/A'

            # --- MODIFIED: Extract abstract from meta description tag --- 
            abstract_meta = article_soup.find('meta', attrs={'name': 'description'})
            abstract = abstract_meta['content'] if abstract_meta and 'content' in abstract_meta.attrs else 'N/A'
            
            pdf_link_tag = article_soup.find('a', class_='c-pdf-button', attrs={'data-track-action': 'Download PDF'})
            pdf_link = f"{base_url}{pdf_link_tag['href']}" if pdf_link_tag and pdf_link_tag.get('href') else 'N/A'

            papers_data.append({
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "pdf_link": pdf_link,
                "page_url": article_url
            })
            scraped_urls.add(article_url)

            # Checkpoint saving for paper data
            if len(papers_data) % checkpoint_interval == 0:
                print(f"  Checkpoint: Saving {len(papers_data)} papers to {output_file}")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(papers_data, f, ensure_ascii=False, indent=4)

            time.sleep(random.random()) # Random delay between 0 and 1 second

        except requests.exceptions.RequestException as e:
            print(f"    Error fetching article page {article_url}: {e}")
        except Exception as e:
            print(f"    Error parsing article page {article_url}: {e}")

    # Final save after all papers are processed
    if papers_data:
        print(f"Finished scraping. Final save of {len(papers_data)} papers to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(papers_data, f, ensure_ascii=False, indent=4)
    else:
        print("No papers were scraped.")

    return papers_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape paper data from Nature Medicine.')
    parser.add_argument('--start_year', type=int, required=True, help='Start year for scraping, e.g., 2022')
    parser.add_argument('--end_year', type=int, required=True, help='End year for scraping, e.g., 2024')
    parser.add_argument('--conference', type=str, required=True, help='Conference name, e.g., CVPR')
    parser.add_argument('--outdir', type=str, required=True, help='Path to save the json file')
    args = parser.parse_args() 

    assert (args.start_year <= args.end_year)
    
    os.makedirs(args.outdir, exist_ok=True)
    output_file = f"{args.outdir}/nature_medicine_papers{args.start_year}-{args.end_year}.json"
    os.makedirs('.temp', exist_ok=True)
    links_saving_file = f'.temp/nature_medicine_article_links{args.start_year}-{args.end_year}.json'

    scrape_nature_medicine_papers(start_year=args.start_year,
        end_year=args.end_year,
        output_file=output_file,
        links_file=links_saving_file)

