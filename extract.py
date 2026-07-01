import gzip
import json
from collections import Counter

def extract_all_companies():
    company_counter = Counter()
    
    print("Reading candidate pool to extract unique corporate names...")
    with open("info/candidates.jsonl", "rt") as f:
        for line in f:
            if not line.strip(): 
                continue
            candidate = json.loads(line)
            
            # Gather from historical career roles
            for job in candidate.get("career_history", []):
                company_name = job.get("company", "").strip().lower()
                if company_name:
                    company_counter[company_name] += 1
                    
    # Print the top 100 most frequent companies so you can review them
    print("\n--- Top 100 Most Frequently Cited Companies ---")
    for comp, count in company_counter.most_common(100):
        print(f"Company: '{comp}' | Occurrences: {count}")

if __name__ == "__main__":
    extract_all_companies()