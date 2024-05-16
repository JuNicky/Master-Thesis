# Example with arguments:
# python evaluate_embeddings_n.py --evaluation_file evaluation_request_12_dossiers_no_requests.json --embedding_provider local_embeddings --embedding_author GroNLP --embedding_function bert-base-dutch-cased --collection_name 12_dossiers_no_requests --vector_db_folder ./vector_stores/12_dossiers_no_requests_chromadb_1024_256_local_embeddings_GroNLP/bert-base-dutch-cased
# python evaluate_embeddings_n.py --evaluation_file evaluation_request_12_dossiers_no_requests.json --embedding_provider local_embeddings --embedding_author meta-llama --embedding_function Meta-Llama-3-8B --collection_name 12_dossiers_no_requests --vector_db_folder ./vector_stores/12_dossiers_no_requests_chromadb_1024_256_local_embeddings_meta-llama/Meta-Llama-3-8B

import json
import os
import pandas as pd
from argparse import ArgumentParser
from common.querier import Querier


def check_relevance(ground_truth, retrieved) -> int:
    """
    Calculates the number of relevant items in the retrieved set.

    Parameters:
    ground_truth (set): The set of ground truth items.
    retrieved (set): The set of retrieved items.

    Returns:
    int: The number of relevant items in the retrieved set.
    """
    return len(retrieved.intersection(ground_truth))


def get_first_n_unique_ids_by_type(
    source_documents: list, n: int, id_type: str
) -> list:
    """
    Extracts the first n unique document IDs from a list of source documents.

    Parameters:
    - source_documents: A list of tuples, where each tuple contains a document object and another value.
    - n: The number of unique document IDs to retrieve.

    Returns:
    A list of the first n unique document IDs.
    """

    if id_type not in ["page_id", "document_id", "dossier_id"]:
        raise ValueError("id_type must be 'page_id', 'document_id', or 'dossier_id'")

    unique_ids = []
    seen = set()
    for doc, _ in source_documents:
        doc_id = doc.metadata[id_type]
        if doc_id not in seen:
            seen.add(doc_id)
            unique_ids.append(doc_id)
        if len(unique_ids) == n:
            break

    return unique_ids


def main():
    parser = ArgumentParser()
    parser.add_argument("-e", "--evaluation_file", type=str)
    parser.add_argument("-p", "--embedding_provider", type=str)
    parser.add_argument("-a", "--embedding_author", type=str)
    parser.add_argument("-f", "--embedding_function", type=str)
    parser.add_argument("-c", "--collection_name", type=str)
    parser.add_argument("-v", "--vector_db_folder", type=str)

    args = parser.parse_args()
    if (
        args.evaluation_file
        and args.embedding_provider
        and args.embedding_author
        and args.embedding_function
        and args.collection_name
        and args.vector_db_folder
    ):
        evaluation_file = args.evaluation_file
        embedding_provider = args.embedding_provider
        embedding_author = args.embedding_author
        embedding_function = args.embedding_function
        complete_embedding_function = f"{embedding_author}/{embedding_function}"
        collection_name = args.collection_name
        vector_db_folder = args.vector_db_folder
    else:
        print(
            "Please provide the source folder of documents, the output folder name, and the database directory.",
            flush=True,
        )
        exit()

    # Selecting the paths
    # path = select_woogle_dump_folders(path='../docs')
    # evaluation_file = "../evaluation/evaluation_request_WoogleDumps_01-04-2024_50_dossiers_no_requests.json"
    # embedding_provider = "local_embeddings"
    # embedding_author = "GroNLP"
    # embedding_function = "bert-base-dutch-cased"
    # complete_embedding_function = f"{embedding_author}/{embedding_function}"
    # collection_name = "WoogleDumps_01-04-2024_12817_dossiers_no_requests_part_1"
    # # vector_db_folder = f"./vector_stores/no_requests_all_parts_chromadb_1024_256_local_embeddings_GroNLP/{embedding_function}"
    # vector_db_folder = f"../vector_stores/no_requests_part_2_chromadb_1024_256_local_embeddings_GroNLP/{embedding_function}"

    with open(f"./evaluation/{evaluation_file}", "r") as file:
        evaluation = json.load(file)

    # If vector store folder does not exist, stop
    if not os.path.exists(vector_db_folder):
        print(
            'There is no vector database for this folder yet. First run "python ingest.py"'
        )
        exit()

    querier = Querier()
    print(f"Making chain for collection: {collection_name}", flush=True)
    print(f"Vector DB folder: {vector_db_folder}", flush=True)
    querier.make_chain(collection_name, vector_db_folder)

    querier_data = querier.vector_store.get()
    querier_data_ids = querier_data["ids"]
    print(f"Length querier data IDs: {len(querier_data_ids)}", flush=True)
    print(f"Max Id in data: {max([int(num) for num in querier_data_ids])}", flush=True)

    print(f"Running algorithm: {complete_embedding_function}", flush=True)

    # Determine file paths
    csv_file_path = f'./evaluation/results/{evaluation_file.split("/")[-1].replace(".json", "")}_{collection_name.replace("_part_1", "")}_{embedding_function}_request.csv'
    json_file_path = f'./evaluation/results/{evaluation_file.split("/")[-1].replace(".json", "")}_{collection_name.replace("_part_1", "")}_{embedding_function}_request_raw.json'
    last_index = -1

    # Check if csv file exists
    csv_file_exists = os.path.exists(csv_file_path)
    csv_file = open(csv_file_path, "a")
    csv_writer = None

    result = pd.DataFrame(
        columns=[
            "page_id",
            "dossier_id",
            "retrieved_page_ids",
            "retrieved_dossier_ids",
            "scores",
            "number_of_correct_dossiers",
            "dossier#1",
            "dossier#2",
            "dossier#3",
            "dossier#4",
            "dossier#5",
            "dossier#6",
            "dossier#7",
            "dossier#8",
            "dossier#9",
            "dossier#10",
            "dossier#11",
            "dossier#12",
            "dossier#13",
            "dossier#14",
            "dossier#15",
            "dossier#16",
            "dossier#17",
            "dossier#18",
            "dossier#19",
            "dossier#20",
        ]
    )

    for index, (key, value) in enumerate(evaluation.items()):
        if index <= last_index:
            print(f"Skipping index {index}", flush=True)
            continue
        if not value.get("pages"):
            print("No pages found in the JSON file", flush=True)
            continue
        if not value.get("documents"):
            print("No documents found in the JSON file", flush=True)
            continue
        if not value.get("dossier"):
            print("No dossiers found in the JSON file", flush=True)
            continue

        # tokenized_query = preprocess_text(key)
        response = querier.ask_question(key)
        source_documents = response["source_documents"]

        retrieved_page_ids = []
        retrieved_dossier_ids = []
        # scores = []
        for doc, _ in source_documents:
            print(doc.metadata)
            retrieved_page_ids.append(doc.metadata["document_id"])
            retrieved_dossier_ids.append(doc.metadata["dossier_id"])
            if len(retrieved_page_ids) == 20:
                break

        print(value)

        # Collect top documents and their scores for the current BM25 algorithm
        new_row = {
            "page_id": "N/A",
            "dossier_id": value["dossier"][0],
            "retrieved_page_ids": ", ".join(retrieved_page_ids),
            "retrieved_dossier_ids": ", ".join(retrieved_dossier_ids),
            "scores": "",
            "number_of_correct_dossiers": retrieved_dossier_ids.count(
                value["dossier"][0]
            ),
            "dossier#1": retrieved_dossier_ids[0] == value["dossier"][0],
            "dossier#2": retrieved_dossier_ids[1] == value["dossier"][0],
            "dossier#3": retrieved_dossier_ids[2] == value["dossier"][0],
            "dossier#4": retrieved_dossier_ids[3] == value["dossier"][0],
            "dossier#5": retrieved_dossier_ids[4] == value["dossier"][0],
            "dossier#6": retrieved_dossier_ids[5] == value["dossier"][0],
            "dossier#7": retrieved_dossier_ids[6] == value["dossier"][0],
            "dossier#8": retrieved_dossier_ids[7] == value["dossier"][0],
            "dossier#9": retrieved_dossier_ids[8] == value["dossier"][0],
            "dossier#10": retrieved_dossier_ids[9] == value["dossier"][0],
            "dossier#11": retrieved_dossier_ids[10] == value["dossier"][0],
            "dossier#12": retrieved_dossier_ids[11] == value["dossier"][0],
            "dossier#13": retrieved_dossier_ids[12] == value["dossier"][0],
            "dossier#14": retrieved_dossier_ids[13] == value["dossier"][0],
            "dossier#15": retrieved_dossier_ids[14] == value["dossier"][0],
            "dossier#16": retrieved_dossier_ids[15] == value["dossier"][0],
            "dossier#17": retrieved_dossier_ids[16] == value["dossier"][0],
            "dossier#18": retrieved_dossier_ids[17] == value["dossier"][0],
            "dossier#19": retrieved_dossier_ids[18] == value["dossier"][0],
            "dossier#20": retrieved_dossier_ids[19] == value["dossier"][0],
        }

        # Append the new value to the DataFrame
        # result.append(new_row, ignore_index=True)
        result.loc[len(result)] = new_row

    loc = f'{evaluation_file.split(".")[0]}_{collection_name}_{embedding_function}_request.csv'
    result.to_csv(f"evaluation/results/{loc}")


if __name__ == "__main__":
    main()
