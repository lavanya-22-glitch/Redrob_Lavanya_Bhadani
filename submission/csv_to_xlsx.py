import pandas as pd
import sys
import os
import argparse

def convert_csv_to_xlsx(input_csv, output_xlsx):
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} does not exist.")
        sys.exit(1)
        
    try:
        print(f"Reading {input_csv}...")
        df = pd.read_csv(input_csv)
        print(f"Writing to {output_xlsx}...")
        # openpyxl is required by pandas for writing xlsx files
        df.to_excel(output_xlsx, index=False)
        print("Conversion complete!")
    except ImportError:
        print("Error: Missing required library.")
        print("Please ensure both 'pandas' and 'openpyxl' are installed.")
        print("You can install them via: pip install pandas openpyxl")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to XLSX")
    parser.add_argument("input", nargs="?", help="Input CSV file (positional)")
    parser.add_argument("output", nargs="?", help="Output XLSX file (positional)")
    parser.add_argument("--file", "-f", help="Input CSV file (optional tag)")
    
    args = parser.parse_args()
    
    input_file = args.file or args.input or "team_Lavanya_Bhadani.csv"
    
    if args.output:
        output_file = args.output
    else:
        output_file = input_file.rsplit('.', 1)[0] + ".xlsx"

    convert_csv_to_xlsx(input_file, output_file)
