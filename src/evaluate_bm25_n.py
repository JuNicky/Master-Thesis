 # Example with arguments:
# python evaluate_bm25.py -a BM25Okapi -c WoogleDumps_01-04-2024_10_dossiers_no_requests_fake_stopwords -d ../docs -e evaluation_request_WoogleDumps_01-04-2024_10_dossiers_no_requests.json
# python evaluate_bm25.py -c 12_dossiers_no_requests -d ../docs_ministries -e evaluation_request_Ministries_12_dossiers_no_requests.json
# python evaluate_bm25.py -c 12_dossiers_no_requests -d /scratch/nju/docs -e evaluation_request_Ministries_12_dossiers_no_requests_keywords.json
# python evaluate_bm25.py -a BM25Okapi -c 12_dossiers_no_requests -d ./docs_ministries -e evaluation_request_12_dossiers_no_requests_keywords.json
import heapq
import json
import nltk
import os
import pandas as pd
import re
from argparse import ArgumentParser
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi, BM25L, BM25Plus

def preprocess_text(text: str, index: int=0, print_progress: bool=True, print_freq:int=100) -> list[str]:
    if print_progress and index and index % print_freq == 0:
        print(f"Processing document {index}", flush=True)
    
    # Initialize stop words and stemmer
    stop_words = set(stopwords.words('dutch'))
    stemmer = PorterStemmer()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove unnecessary whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stop words and stem
    return [stemmer.stem(word) for word in tokens if word not in stop_words]

def tokenize(text) -> list[str]:
    # Check if text is of type string
    if not isinstance(text, str):
        return []
    # Tokenize the text
    return word_tokenize(text)
    
def check_relevance(ground_truth, retrieved) -> int:
    # Check if the retrieved documents are relevant
    return len(retrieved.intersection(ground_truth))

def run_bm25(woo_data, bm25, evaluation, evaluation_file, content_folder_name):
    # Check if chunks are present in the data
    print(f"Running algorithm: {bm25.__class__.__name__}", flush=True)

    # Determine file paths
    csv_file_path = f'./evaluation/results/{evaluation_file.split(".")[0]}_{content_folder_name}_{bm25.__class__.__name__}_request.csv'
    json_file_path = f'./evaluation/results/{evaluation_file.split(".")[0]}_{content_folder_name}_{bm25.__class__.__name__}_request_raw.json'
    last_index = -1
    
    # Check if csv file exists
    csv_file_exists = os.path.exists(csv_file_path)
    csv_file = open(csv_file_path, 'a')
    csv_writer = None
    result = pd.DataFrame(columns=["page_id", "dossier_id", "retrieved_page_ids", "retrieved_dossier_ids", "scores", "number_of_correct_dossiers", "dossier#1", "dossier#2", "dossier#3", "dossier#4", "dossier#5", "dossier#6", "dossier#7", "dossier#8", "dossier#9", "dossier#10", "dossier#11", "dossier#12", "dossier#13", "dossier#14", "dossier#15", "dossier#16", "dossier#17", "dossier#18", "dossier#19", "dossier#20"])
    
    for index, (key, value) in enumerate(evaluation.items()):
        if index <= last_index:
            print(f"Skipping index {index}", flush=True)
            continue
        results_raw = {}
        if not value.get('pages'):
            print("No pages found in the JSON file", flush=True)
            continue
        if not value.get('documents'):
            print("No documents found in the JSON file", flush=True)
            continue
        if not value.get('dossier'):
            print("No dossiers found in the JSON file", flush=True)
            continue

        # tokenized_query = preprocess_text(key)
        tokenized_query = tokenize(key)
        
        doc_scores = bm25.get_scores(tokenized_query)
        
        print(doc_scores)
            
        n_pages_result = heapq.nlargest(20, range(len(doc_scores)), key=doc_scores.__getitem__)
        retrieved_page_ids = []
        retrieved_dossier_ids = []
        # scores = []
        for i in n_pages_result:
            retrieved_page_ids.append(woo_data['document_id'][i])
            retrieved_dossier_ids.append(woo_data['dossier_id'][i])
            
        print(value)
        
        # Collect top documents and their scores for the current BM25 algorithm
        new_row = {
            'page_id': "N/A",
            'dossier_id': value['dossier'][0],
            'retrieved_page_ids': ", ".join(retrieved_page_ids),
            'retrieved_dossier_ids': ", ".join(retrieved_dossier_ids),
            'scores': "",
            'number_of_correct_dossiers': retrieved_dossier_ids.count(value['dossier'][0]),
            'dossier#1': retrieved_dossier_ids[0] == value['dossier'][0],
            'dossier#2': retrieved_dossier_ids[1] == value['dossier'][0],
            'dossier#3': retrieved_dossier_ids[2] == value['dossier'][0],
            'dossier#4': retrieved_dossier_ids[3] == value['dossier'][0],
            'dossier#5': retrieved_dossier_ids[4] == value['dossier'][0],
            'dossier#6': retrieved_dossier_ids[5] == value['dossier'][0],
            'dossier#7': retrieved_dossier_ids[6] == value['dossier'][0],
            'dossier#8': retrieved_dossier_ids[7] == value['dossier'][0],
            'dossier#9': retrieved_dossier_ids[8] == value['dossier'][0],
            'dossier#10': retrieved_dossier_ids[9] == value['dossier'][0],
            'dossier#11': retrieved_dossier_ids[10] == value['dossier'][0],
            'dossier#12': retrieved_dossier_ids[11] == value['dossier'][0],
            'dossier#13': retrieved_dossier_ids[12] == value['dossier'][0],
            'dossier#14': retrieved_dossier_ids[13] == value['dossier'][0],
            'dossier#15': retrieved_dossier_ids[14] == value['dossier'][0],
            'dossier#16': retrieved_dossier_ids[15] == value['dossier'][0],
            'dossier#17': retrieved_dossier_ids[16] == value['dossier'][0],
            'dossier#18': retrieved_dossier_ids[17] == value['dossier'][0],
            'dossier#19': retrieved_dossier_ids[18] == value['dossier'][0],
            'dossier#20': retrieved_dossier_ids[19] == value['dossier'][0]
        }
        
        # Append the new value to the DataFrame
        # result.append(new_row, ignore_index=True)
        result.loc[len(result)] = new_row
            
    loc = f'{evaluation_file.split(".")[0]}_{content_folder_name}_{bm25.__class__.__name__}_request.csv'
    result.to_csv(f"evaluation/results/{loc}")


def main():
    # If necessary, download the NLTK resources
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    
    print("Successfully downloaded the NLTK resources.", flush=True)
    
    parser = ArgumentParser()
    parser.add_argument('-a', '--algorithm', type=str)
    parser.add_argument('-c', '--content_folder_name', type=str)
    parser.add_argument('-d', '--documents_directory', type=str)
    parser.add_argument('-e', '--evaluation_file', type=str)
    parser.add_argument('-r', '--retrieve_whole_document', type=bool, default=False)
    
    args = parser.parse_args()
    if args.content_folder_name and args.documents_directory and args.evaluation_file:
        content_folder_name = args.content_folder_name
        documents_directory = args.documents_directory
        evaluation_file = args.evaluation_file
        retrieve_whole_document = args.retrieve_whole_document
        if args.algorithm in ["BM25Okapi", "BM25L", "BM25Plus"]:
            algorithm = args.algorithm
        else:
            algorithm = "all"
    else:
        print("Please provide the source folder of documents, the output folder name, and the database directory.", flush=True)
        exit()
    print(f"Source folder of documents: {content_folder_name}", flush=True)

    
    # Selecting the paths
    file_name = "woo_merged.csv.gz"
    input_path = f'{documents_directory}/{content_folder_name}/{file_name}'
    evaluation_path = f'./evaluation/{evaluation_file}'

    woo_data = pd.read_csv(input_path, compression='gzip')
    
    # Preprocess woo data, merge all the documents into one entry
    if retrieve_whole_document:
        woo_data = woo_data.groupby('document_id').agg({
            'bodyText': lambda x: ' '.join(x.astype(str)),
            'page_id': 'first',
            'dossier_id': 'first',
        }).reset_index()

    # Generate corpus, which is a list of all the words per document
    corpus = woo_data['bodyText'].tolist()

    print(f"Number of documents in corpus: {len(corpus)}", flush=True)

    # Do preprocessing for echt document
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    
    with open(evaluation_path, 'r') as file:
        evaluation = json.load(file)
        
    print(f"Number of documents in evaluation: {len(evaluation)}", flush=True)
    
    if algorithm == "BM25Okapi" or algorithm == "all":
        print("Starting BM25Okapi", flush=True)
        bm25okapi = BM25Okapi(tokenized_corpus)
        run_bm25(woo_data, bm25okapi, evaluation, evaluation_file, content_folder_name)
        print("BM25Okapi done", flush=True)
    if algorithm == "BM25L" or algorithm == "all":
        print("Starting BM25L", flush=True)
        bm25l = BM25L(tokenized_corpus)
        run_bm25(woo_data, bm25l, evaluation, evaluation_file, content_folder_name)
        print("BM25L done", flush=True)
    if algorithm == "BM25Plus" or algorithm == "all":
        print("Starting BM25Plus", flush=True)
        bm25plus = BM25Plus(tokenized_corpus)
        run_bm25(woo_data, bm25plus, evaluation, evaluation_file, content_folder_name)
        print("BM25Plus done", flush=True)

if __name__ == "__main__":
    print("Starting the program...", flush=True)
    main()