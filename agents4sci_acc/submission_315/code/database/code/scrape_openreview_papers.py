
import openreview
import openreview.tools
import json
import os
import argparse


# --- Configuration ---
# Replace with your OpenReview username and password if you need to access private information.
# For public papers, you might not need credentials.
# It's recommended to use environment variables or a secure configuration management system
# for credentials in a production environment.
# OPENREVIEW_USERNAME = os.getenv("OPENREVIEW_USERNAME")
# OPENREVIEW_PASSWORD = os.getenv("OPENREVIEW_PASSWORD")

def get_accepted_papers(ICML_GROUP_ID):
    client = None
    try:
        client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')

    except openreview.OpenReviewException as e:
        print(f"Error connecting to OpenReview: {e}")
        print("Please check your username, password, and baseurl.")
        return []

    print(f"Fetching submissions for group: {ICML_GROUP_ID}")

    accepted_papers = []
    try:
        # Fetch all notes (submissions) associated with the Conference group
        # We'll filter by decision later
        all_submissions = client.get_all_notes(content={'venueid': ICML_GROUP_ID},details = 'directReplies')

        for submission in all_submissions:
            title = submission.content.get('title', {}).get('value','')
            authors = submission.content.get('authors',  {}).get('value',[])
            abstract = submission.content.get('abstract',  {}).get('value','')
            pdf_link = submission.content.get('pdf',  {}).get('value','') # This might be a relative path or just the filename
            page_url = f"https://openreview.net/forum?id={submission.id}"

            # Check for decision. The exact key and values for decisions need to be confirmed.
            # Assuming 'decision' is a key in content and its value indicates acceptance type.
            # Common decision values might be "Accept (Oral)", "Accept (Poster)", "Accept (Spotlight)", etc.
            if submission.details is not None:
                decision = [reply['content']['decision']['value'] for reply in submission.details.get('directReplies',[]) if 'decision' in reply['content'] ]
                if len(decision) > 0:
                    decision = decision[0]
                else:
                    decision =  'N/A'
            else:
                decision =  'N/A'
            if any(['Camera_Ready' in i for i in  submission.invitations]):
                decision = "Accept"


            # Filter for oral, highlight, and poster papers
            # These strings are assumptions and might need adjustment based on actual OpenReview data.
            if "Accept" in decision:
                paper_info = {
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "pdf_link": pdf_link,
                    "page_url": page_url
                }
                accepted_papers.append(paper_info)
                print(f"Found accepted paper: {title} ({decision})")

    except Exception as e:
        print(f"An error occurred while fetching submissions: {e}")

    return accepted_papers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape paper data from CVF Open Access.')
    parser.add_argument('--year', type=int, required=True, help='Conference year, e.g., 2024')
    parser.add_argument('--conference', type=str, required=True, help='Conference name, e.g., ICLR')
    parser.add_argument('--outdir', type=str, required=True, help='Path to save the json file')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    ICML_GROUP_ID = f"{args.conference}.cc/{args.year}/Conference"
    OUTPUT_FILE = f"{args.outdir}/{args.conference}_{args.year}_accepted_papers.json"

    print(f"Starting to fetch {args.conference} {args.year} accepted papers...")
    papers = get_accepted_papers(ICML_GROUP_ID)

    if papers:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(papers, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved {len(papers)} accepted papers to {OUTPUT_FILE}")
    else:
        print("No accepted papers found or an error occurred.")
