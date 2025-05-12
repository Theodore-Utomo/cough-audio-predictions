import os
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

# This code checks to see what kinds of audio files exist within our kaggle dataset

def list_file_types(folder_path):
    file_types = Counter()

    # Iterate over all entries in the given folder
    for entry in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry)
        if os.path.isfile(full_path):
            # Extracts file extension as lowercase
            _, ext = os.path.splitext(entry)
            if ext:
                file_types[ext.lower()] += 1
            else:
                file_types['[no extension]'] += 1
    total = 0
    for file_type, count in file_types.items():
        if file_type != ".json" and file_type != ".csv":
            total += count
    print(total)
    return file_types

if __name__ == "__main__":
    folder = os.environ.get("INPUT_FOLDER")
    types = list_file_types(folder)

    print("File type counts in the folder:")
    for file_type, count in types.items():
        print(f"{file_type}: {count}")
