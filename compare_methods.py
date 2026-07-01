import pandas as pd
from pathlib import Path

def main():
    root = Path(__file__).parent
    file2 = root / "team_redrob_challenge_rrf_v3.csv"
    file1 = root / "team_redrob_challenge_rrf.csv"
    
    if not file1.exists() or not file2.exists():
        print("Error: Both CSV files must exist to compare them.")
        return

    # Load data
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Rename columns to distinguish
    df1 = df1.rename(columns={"rank": "rank_m1", "score": "score_m1"})
    df2 = df2.rename(columns={"rank": "rank_rrf", "score": "score_rrf"})
    
    # Merge on candidate_id
    merged = pd.merge(df1[["candidate_id", "rank_m1"]], df2[["candidate_id", "rank_rrf"]], on="candidate_id", how="outer")
    
    overlap = merged.dropna()
    overlap_count = len(overlap)
    
    print("=" * 60)
    print("METHOD 1 vs. RRF (METHOD 2) COMPARISON")
    print("=" * 60)
    print(f"Top 100 Overlap: {overlap_count}% of candidates are in both lists.\n")
    
    # Calculate rank change
    # Positive change means RRF ranked them HIGHER (smaller rank number)
    overlap["rank_change"] = overlap["rank_m1"] - overlap["rank_rrf"]
    
    print("--- Top 10 in RRF ---")
    top_rrf = df2.head(10).merge(df1[["candidate_id", "rank_m1"]], on="candidate_id", how="left")
    for _, row in top_rrf.iterrows():
        old_rank = f"#{int(row['rank_m1'])}" if pd.notna(row['rank_m1']) else "Not in top 100"
        print(f"RRF #{row['rank_rrf']:<3} | {row['candidate_id']} | M1 Rank: {old_rank}")
        
    print("\n--- Biggest Gainers (in both top 100s) ---")
    gainers = overlap.sort_values("rank_change", ascending=False).head(5)
    for _, row in gainers.iterrows():
        print(f"{row['candidate_id']}: Rank #{int(row['rank_m1'])} -> #{int(row['rank_rrf'])} (+{int(row['rank_change'])} spots)")

    print("\n--- Dropped from Top 100 (in M1, but not in RRF) ---")
    dropped = merged[merged["rank_rrf"].isna()].sort_values("rank_m1")
    print(f"Total dropped: {len(dropped)}")
    if len(dropped) > 0:
        print("Top 5 highest ranked in M1 that were dropped:")
        for _, row in dropped.head(5).iterrows():
            print(f"  {row['candidate_id']} (was M1 #{int(row['rank_m1'])})")

    print("\n--- New Entrants to Top 100 (in RRF, but not in M1) ---")
    new_entrants = merged[merged["rank_m1"].isna()].sort_values("rank_rrf")
    print(f"Total new entrants: {len(new_entrants)}")
    if len(new_entrants) > 0:
        print("Top 5 highest ranked in RRF that are new:")
        for _, row in new_entrants.head(5).iterrows():
            print(f"  {row['candidate_id']} (now RRF #{int(row['rank_rrf'])})")
            
    print("=" * 60)

if __name__ == "__main__":
    main()
