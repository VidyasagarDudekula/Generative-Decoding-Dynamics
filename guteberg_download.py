import pandas as pd
import requests
import re
import csv
import time
import os
from config import ModelArgs


config = ModelArgs()


def strip_gutenberg_headers(text):
    start_markers = [
        r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK",
        r"\*\*\* START OF THE PROJECT GUTENBERG",
        r"\*\*\*START OF THE PROJECT GUTENBERG"
    ]
    end_markers = [
        r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK",
        r"\*\*\* END OF THE PROJECT GUTENBERG",
        r"\*\*\*END OF THE PROJECT GUTENBERG"
    ]

    start_pos = 0
    for marker in start_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            start_pos = match.end()
            break
            
    end_pos = len(text)
    for marker in end_markers:
        match = re.search(marker, text, re.IGNORECASE)
        if match:
            end_pos = match.start()
            break
    
    return text[start_pos:end_pos]

def clean_text_regex(text):
    if not text:
        return ""
    

    text = re.sub(r'[\r\n\t]+', ' ', text)
    

    replacements = {'“': '"', '”': '"', '’': "'", '‘': "'", '—': '-'}
    for char, rep in replacements.items():
        text = text.replace(char, rep)
        
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


input_csv = 'gutenberg_metadata.csv'
output_csv = config.data_file_path


if not os.path.exists(input_csv):
    print(f"Error: {input_csv} not found.")
    exit()

df_metadata = pd.read_csv(input_csv)
print(f"Found {len(df_metadata)} books to process.")


with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    writer.writerow(['Title', 'Author', 'Link', 'ID', 'Bookshelf', 'Text'])
    
    for index, row in df_metadata.iterrows():
        try:
            book_id = str(row['Link']).strip('/').split('/')[-1]
            title = row.get('Title', 'Unknown')
            
            download_url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
            
            response = requests.get(download_url, timeout=10)
            
            if response.status_code == 200:
                response.encoding = 'utf-8'
                
                raw_text = response.text
                content = strip_gutenberg_headers(raw_text)
                cleaned_text = clean_text_regex(content)
                
                data_row = [
                    title,
                    row.get('Author'),
                    row.get('Link'),
                    book_id,
                    row.get('Bookshelf'),
                    cleaned_text
                ]
                
                writer.writerow(data_row)
                
                del raw_text
                del content
                del cleaned_text
                
                print(f"[{index+1}/{len(df_metadata)}] Saved: {title} (ID: {book_id})")
                
            else:
                print(f"[{index+1}/{len(df_metadata)}] Failed (Status {response.status_code}): {title}")

            time.sleep(0.5)

        except Exception as e:
            print(f"[{index+1}/{len(df_metadata)}] Error processing ID {book_id}: {e}")
            continue

print("Processing complete.")