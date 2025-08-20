import os
import re
import argparse
from tqdm import tqdm
import pandas as pd
from utils import get_file_hash


def add_args(parser: argparse.ArgumentParser):
    pass

def get_metadata(**kwargs):
    # make your own metadata
    original_data_dir = "/mnt/data/xinghui/space-nvs/sat_splits/train/models/"  # Change to your original_data directory
    obj_files = [f for f in os.listdir(original_data_dir) if f.endswith('.obj')]

    metadata = {"sha256": [], "file_identifier": [], "aesthetic_score": [], "captions": []}
    for idx, obj_file in enumerate(obj_files):
        obj_path = os.path.join(original_data_dir, obj_file)
        sha256 = get_file_hash(obj_path)

        metadata["sha256"].append(sha256)
        metadata["file_identifier"].append(obj_path)

        # Empty for this dataset
        metadata["aesthetic_score"].append(0)
        metadata["captions"].append("")

    metadata = pd.DataFrame(metadata)
    return metadata
        

def download(metadata, output_dir, **kwargs):
    downloaded = {}
    metadata = metadata.set_index("file_identifier")

    for k, sha256 in zip(metadata.index, metadata["sha256"]):

        downloaded[sha256] = k

    return pd.DataFrame(downloaded.items(), columns=['sha256', 'local_path'])


def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects') -> pd.DataFrame:
    import os
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    
    # load metadata
    metadata = metadata.to_dict('records')

    # processing objects
    records = []
    max_workers = max_workers or os.cpu_count()
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor, \
            tqdm(total=len(metadata), desc=desc) as pbar:
            def worker(metadatum):
                try:
                    local_path = metadatum['local_path']
                    sha256 = metadatum['sha256']
                    file = local_path
                    record = func(file, sha256)
                    if record is not None:
                        records.append(record)
                    pbar.update()
                except Exception as e:
                    print(f"Error processing object {sha256}: {e}")
                    pbar.update()
            
            executor.map(worker, metadata)
            executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")
        
    return pd.DataFrame.from_records(records)
