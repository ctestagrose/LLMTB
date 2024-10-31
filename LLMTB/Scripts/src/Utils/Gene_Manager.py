import logging
import json


class GeneManager:
    def __init__(self, gene_file):
        self.genes = self.load_genes(gene_file)

    def load_genes(self, gene_file):
        with open(gene_file, 'r') as f:
            return json.load(f)

    def read_fasta_file_genes(self, file_path, target, filter_xs=False):
        sorted_genes = {key: sorted(value) for key, value in self.genes.items()}
        gene_sequences = {gene.lower(): None for gene in sorted_genes[target]}

        try:
            with open(file_path, 'r') as file:
                sequence = ''
                gene_name = ''
                for line in file:
                    if "<<P>>" in line:
                        return None
                    if not line.startswith('>'):
                        sequence += line.strip().upper()
                    elif line.startswith('>') and sequence != '':
                        clean_sequence = sequence.replace('\n', '').upper()
                        if clean_sequence and gene_name in gene_sequences:
                            gene_sequences[gene_name] = clean_sequence
                        sequence = ''
                    if line.startswith('>'):
                        gene_name = line.split()[-1].strip("]").lower()
                        gene_name_o = line.split("|")[-3].strip("]").lower()
                        if gene_name_o.lower() in gene_sequences:
                            gene_name = gene_name_o
                clean_sequence = sequence.replace('\n', '').upper()
                if clean_sequence and gene_name in gene_sequences:
                    gene_sequences[gene_name] = clean_sequence
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")

        ordered_sequences = []
        genes_in_isolate = []
        missing = []

        for gene in sorted_genes[target]:
            gene_lower = gene.lower()
            sequence = gene_sequences.get(gene_lower, None)
            if sequence is None or "X" in sequence:
                missing.append(gene)
                ordered_sequences.append("XXXXXXXXXXXXXXXXXX")
                genes_in_isolate.append(gene)
            else:
                ordered_sequences.append(sequence)
                genes_in_isolate.append(gene)
        
        print(f"MISSING {len(missing)} out of {len(sorted_genes[target])}")

        if filter_xs:
            filtered_sequences = []
            filtered_genes = []

            for seq, gene in zip(ordered_sequences, genes_in_isolate):
                if "X" not in seq:
                    filtered_sequences.append(seq)
                    filtered_genes.append(gene)

            ordered_sequences = filtered_sequences
            genes_in_isolate = filtered_genes

            print(f"Number of ordered sequences: {len(ordered_sequences)}")
            print(f"Genes in isolate: {genes_in_isolate}")

        if not ordered_sequences:
            logging.warning(f"No valid sequences found in file {file_path}")

        return ordered_sequences, genes_in_isolate

    def read_fasta_file_genes_all(self, file_path, target=None, filter_xs=False):
        exclude_hypothetical = False
        filter_xs=False

        sorted_genes = set()
        
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    if line.startswith('>'):
                        gene_name = line.split("|")[-3].strip("]").lower()
                        if exclude_hypothetical and "hypothetical" in gene_name:
                            continue
                        sorted_genes.add(gene_name)
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            return [], []

        sorted_genes = sorted(sorted_genes)

        gene_sequences = {gene: None for gene in sorted_genes}

        try:
            with open(file_path, 'r') as file:
                sequence = ''
                gene_name = None
                for line in file:
                    line = line.strip()
                    if line.startswith('>'):
                        if sequence and gene_name and gene_name in gene_sequences:
                            gene_sequences[gene_name] = sequence.upper()
                        sequence = ''
                        gene_name = line.split("|")[-3].strip("]").lower()
                        if exclude_hypothetical and "hypothetical" in gene_name:
                            gene_name = None  # Skip this gene entirely
                    else:
                        sequence += line.upper()

                if sequence and gene_name and gene_name in gene_sequences:
                    gene_sequences[gene_name] = sequence.upper()
                    
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")

        ordered_sequences = []
        genes_in_isolate = []
        missing = []
    
        for gene in sorted_genes:
            sequence = gene_sequences.get(gene, None)
            if sequence is None or "X" in sequence:
                missing.append(gene)
                ordered_sequences.append("XXXXXXXXXXXXXXXXXX")
            else:
                ordered_sequences.append(sequence)
            genes_in_isolate.append(gene)
    
        print(f"MISSING {len(missing)}")
    
        if filter_xs:
            filtered_sequences = [seq for seq in ordered_sequences if "X" not in seq]
            filtered_genes = [gene for seq, gene in zip(ordered_sequences, genes_in_isolate) if "X" not in seq]
    
            ordered_sequences = filtered_sequences
            genes_in_isolate = filtered_genes

            print(f"Number of ordered sequences: {len(ordered_sequences)}")
            print(f"Genes in isolate: {genes_in_isolate}")
    
        if not ordered_sequences:
            logging.warning(f"No valid sequences found in file {file_path}")
    
        print(len(ordered_sequences))
    
        return ordered_sequences, genes_in_isolate

