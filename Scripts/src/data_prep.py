import json
import math
from tqdm import tqdm

class DataPreparer:
    def __init__(self, gene_manager, sequence_processor, tokenizer_manager, args):
        # Initialize DataPreparer
        self.gene_manager = gene_manager
        self.sequence_processor = sequence_processor
        self.tokenizer_manager = tokenizer_manager
        self.args = args

    def prep_data(self, fasta_files, target_file, target_format, mode):
        
        # Initialize the lists to hold real sequences, labels, and seq_ids
        sequences = []
        labels = []
        seq_ids = []
        seq_id_counter = 0
        genes_list = []

        # Read in the cryptic targets
        with open(target_file, "r") as f:
            targets = json.load(f)

        if mode == "Evaluate" or mode == "Train":
            split_dict = {}
            unsplit_dict = {}
            for item in targets:
                if "." in item:
                    for err in item.split("."):
                        temp_item = targets[item]
                        split_dict[err] = temp_item
                else:
                    split_dict[item] = targets[item]
                    unsplit_dict[item] = targets[item]
            
            targets = split_dict

        # Iterate through the fasta files
        for fasta_file in tqdm(fasta_files, desc=f"Processing FASTA files for {target_format}...", unit="files"):
            # Get isolate number
            isolate = fasta_file.split("/")[-1].split(".")[0]
            if "_" in isolate:
                isolate = isolate.split("_")[0]
                target_format = self.args.antibiotic
            if isolate in targets:
                label = targets[isolate][target_format]
                
                # Check for valid labels (ignoring isolates with NaN values in any label)
                if target_format and not math.isnan(label):
                    clean_sequence, genes_in_isolate = self.gene_manager.read_fasta_file_genes(fasta_file, target_format)
        
                    # If there is a sequence, append; else, print no valid sequences
                    if clean_sequence:
                        for seq in clean_sequence:
                            if "<P>" in clean_sequence:
                                print(clean_sequence)
                        sequences.append(clean_sequence)

                        labels.append(label)
                        seq_ids.append(isolate)
                        seq_id_counter += 1
                        genes_list.append(genes_in_isolate)
                    else:
                        print(f"Skipping file {fasta_file} as it contains no valid sequences.")

        zipped_data = list(zip(sequences, labels, seq_ids, genes_list))

        return zipped_data


