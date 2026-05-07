# Explorer Agent: Extracts drug-symptom associations from Reddit oncology discussions using PRAW

import praw
import json
import os
from datetime import datetime
from typing import List, Dict
import time
import re

ONCOLOGY_DRUGS = [
    'keytruda', 'pembrolizumab', 'opdivo', 'nivolumab', 'tecentriq', 'atezolizumab',
    'yervoy', 'ipilimumab', 'imfinzi', 'durvalumab', 'bavencio', 'avelumab',
    'herceptin', 'trastuzumab', 'avastin', 'bevacizumab', 'rituxan', 'rituximab',
    'doxorubicin', 'adriamycin', 'carboplatin', 'cisplatin', 'taxol', 'paclitaxel',
    'taxotere', 'docetaxel', 'gemzar', 'gemcitabine', '5-fu', 'fluorouracil',
    'tamoxifen', 'letrozole', 'femara', 'arimidex', 'anastrozole',
    'ibrance', 'palbociclib', 'kisqali', 'ribociclib', 'verzenio', 'abemaciclib',
    'lynparza', 'olaparib', 'zejula', 'niraparib', 'rubraca', 'rucaparib',
    'tagrisso', 'osimertinib', 'tarceva', 'erlotinib', 'iressa', 'gefitinib',
    'alimta', 'pemetrexed', 'abraxane', 'xeloda', 'capecitabine',
    'revlimid', 'lenalidomide', 'velcade', 'bortezomib', 'kyprolis', 'carfilzomib',
    'gleevec', 'imatinib', 'sprycel', 'dasatinib', 'tasigna', 'nilotinib',
    'yescarta', 'axicabtagene ciloleucel',
    'kymriah', 'tisagenlecleucel',
    'breyanzi', 'lisocabtagene maraleucel',
    'abecma', 'idecabtagene vicleucel',
    'carvykti', 'ciltacabtagene autoleucel'
]

CANCER_SUBREDDITS = [
    'cancer', 'breastcancer', 'lungcancer', 'prostatecancer', 'coloncancer',
    'leukemia', 'lymphoma', 'braincancer', 'ovariancancer', 'pancreaticcancer',
    'melanoma', 'thyroidcancer', 'kidneycancer', 'bladder_cancer', 'testicularcancer',
    'multiplemyeloma', 'esophagealcancer', 'stomachcancer', 'oralcancer', 'sarcoma',
    'chemotherapy', 'radiation', 'immunotherapy', 'carttherapy', 'celltherapy',
    'cancersurvivors', 'cancercaregivers', 'supportgroups', 'patients', 'chronicillness',
    'AskDocs', 'medical_advice'
]

OUTPUT_DIR = "reddit_data"

class SimpleRedditScraper:
    def __init__(self):
        self.reddit = praw.Reddit(
            client_id="",
            client_secret="",
            user_agent="PharmacovigillanceResearch/1.0"
        )
        
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        print("Initialized Reddit scraper in read-only mode")
    
    def contains_drug_mention(self, text: str) -> bool:
        text_lower = text.lower()
        return any(drug in text_lower for drug in ONCOLOGY_DRUGS)
    
    def scrape_post(self, post_url: str = None, post_id: str = None) -> Dict:
        try:
            if post_url:
                submission = self.reddit.submission(url=post_url)
            elif post_id:
                submission = self.reddit.submission(id=post_id)
            else:
                raise ValueError("Either post_url or post_id must be provided")
            
            post_data = {
                'id': submission.id,
                'title': submission.title,
                'text': submission.selftext,
                'author': str(submission.author) if submission.author else '[deleted]',
                'subreddit': submission.subreddit.display_name,
                'created_utc': datetime.fromtimestamp(submission.created_utc).isoformat(),
                'score': submission.score,
                'url': submission.url,
                'num_comments': submission.num_comments,
                'comments': []
            }
            
            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list():
                if comment.body and comment.body != '[deleted]' and comment.body != '[removed]':
                    comment_data = {
                        'id': comment.id,
                        'text': comment.body,
                        'author': str(comment.author) if comment.author else '[deleted]',
                        'created_utc': datetime.fromtimestamp(comment.created_utc).isoformat(),
                        'score': comment.score
                    }
                    post_data['comments'].append(comment_data)
            
            return post_data
            
        except Exception as e:
            print(f"Error scraping post: {e}")
            return None
    
    def scrape_subreddit(self, subreddit_name: str, 
                        sort_by: str = 'hot', 
                        time_filter: str = 'month',
                        limit: int = 100) -> List[Dict]:
        print(f"\nScraping r/{subreddit_name} ({sort_by} posts)...")
        subreddit = self.reddit.subreddit(subreddit_name)
        
        posts_data = []
        
        if sort_by == 'hot':
            posts = subreddit.hot(limit=limit)
        elif sort_by == 'top':
            posts = subreddit.top(time_filter=time_filter, limit=limit)
        elif sort_by == 'new':
            posts = subreddit.new(limit=limit)
        elif sort_by == 'random':
            posts = subreddit.random_rising(limit=limit)
        else:
            posts = subreddit.hot(limit=limit)
        
        for submission in posts:
            full_text = submission.title + " " + submission.selftext
            if self.contains_drug_mention(full_text):
                print(f"  Processing: {submission.title[:60]}...")
                
                post_data = {
                    'id': submission.id,
                    'title': submission.title,
                    'text': submission.selftext,
                    'author': str(submission.author) if submission.author else '[deleted]',
                    'subreddit': subreddit_name,
                    'created_utc': datetime.fromtimestamp(submission.created_utc).isoformat(),
                    'score': submission.score,
                    'url': submission.url,
                    'num_comments': submission.num_comments,
                    'comments': []
                }
                
                if submission.num_comments > 0:
                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments.list()[:50]:
                        if comment.body and comment.body != '[deleted]':
                            if self.contains_drug_mention(comment.body):
                                comment_data = {
                                    'id': comment.id,
                                    'text': comment.body,
                                    'author': str(comment.author) if comment.author else '[deleted]',
                                    'created_utc': datetime.fromtimestamp(comment.created_utc).isoformat(),
                                    'score': comment.score
                                }
                                post_data['comments'].append(comment_data)
                
                posts_data.append(post_data)
                time.sleep(1)
        
        print(f"  Found {len(posts_data)} posts with drug mentions")
        return posts_data

def simple_extraction(post_data: Dict) -> Dict:
    extractions = []
    
    full_text = post_data['title'] + " " + post_data['text']
    for comment in post_data.get('comments', []):
        full_text += " " + comment['text']
    
    full_text_lower = full_text.lower()
    
    side_effect_patterns = [
        (r'(started|began|on) (\w+).{0,50}(experiencing|having|getting|developed) ([^.]+)', 0.8),
        (r'(\w+) (caused|causing|gives me|gave me) ([^.]+)', 0.9),
        (r'side effects?.{0,30}(\w+).{0,30}include ([^.]+)', 0.7),
        (r'since.{0,20}(\w+).{0,30}(i\'ve been|i have been|i\'ve had) ([^.]+)', 0.85),
    ]
    
    community_metric = post_data.get("score", 0) + post_data.get("num_comments", 0)
    
    for drug in ONCOLOGY_DRUGS:
        if drug in full_text_lower:
            sentences = re.split(r'[.!?]', full_text)
            for sentence in sentences:
                if drug in sentence.lower():
                    symptom_words = ['fatigue', 'tired', 'nausea', 'vomiting', 'rash', 'pain', 
                                   'diarrhea', 'fever', 'headache', 'neuropathy', 'tingling',
                                   'numbness', 'hair loss', 'weight', 'appetite', 'taste']
                    
                    for symptom in symptom_words:
                        if symptom in sentence.lower():
                            extractions.append({
                                "drug": drug,
                                "drug_canonical": drug,
                                "side_effect": symptom,
                                "side_effect_medical": symptom,
                                "temporal_weight": 0.5,
                                "age": 20,
                                "severity": "not specified",
                                "quote": sentence[:200],
                                "confidence": 0.6,
                                "community_metric": community_metric
                            })
    
    json_path = os.path.join(OUTPUT_DIR, f"{post_data['id']}.json")
    save_data = {
        "post_id": post_data['id'],
        "subreddit": post_data['subreddit'], 
        "date": post_data['created_utc'],
        "url": post_data['url'],
        "extractions": extractions,
        "summary": f"Found {len(extractions)} potential drug-side effect pairs (rule-based extraction)",
        "analyzed_at": datetime.now().isoformat()
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2)
    
    return {
        "extractions": extractions,
        "has_drug_mentions": len(extractions) > 0,
        "summary": save_data["summary"]
    }

def main():
    print("Reddit Pharmacovigilance Data Collector")
    print("="*60)
    
    scraper = SimpleRedditScraper()
    
    print("\n1. Collecting posts from cancer subreddits...")
    
    for subreddit in CANCER_SUBREDDITS:
        all_posts = []
        try:
            posts = scraper.scrape_subreddit(
                subreddit_name=subreddit,
                sort_by='new',
                time_filter='year',
                limit=100
            )
            all_posts.extend(posts)
        except Exception as e:
            print(f"  Error with r/{subreddit}: {e}")
            continue
    
        print(f"\n2. Analyzing {len(all_posts)} posts for drug-side effect pairs...")
    
        for post in all_posts:
            try:
                result = simple_extraction(post)
                
                if result['has_drug_mentions']:
                    print(f"  ✓ Found {len(result['extractions'])} pairs in post {post['id']}")
            except Exception as e:
                print(f"  Error analyzing post {post['id']}: {e}")
    
    json_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.json')]
    total_extractions = 0
    
    for json_file in json_files:
        with open(os.path.join(OUTPUT_DIR, json_file), 'r') as f:
            data = json.load(f)
            total_extractions += len(data.get('extractions', []))
    
    print(f"\n4. Summary:")
    print(f"   - Posts processed: {len(json_files)}")
    print(f"   - Total extractions: {total_extractions}")

if __name__ == "__main__":
    main()