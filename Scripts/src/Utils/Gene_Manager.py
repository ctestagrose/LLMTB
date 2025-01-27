import logging
import json


class GeneManager:
    def __init__(self, gene_file):
        self.genes = self.load_genes(gene_file)

    def load_genes(self, gene_file):
        """
        Loads the json file with the genes of interest

        :param gene_file: json file containing a dictionary of lists ({antibiotic:[genes...]})
        :return the loaded dictionary
        """
        with open(gene_file, 'r') as f:
            return json.load(f)

    def read_fasta_file_genes(self, file_path, target, filter_xs=False, sort_by_genes=True, use_gene_file=True,
                              method="separate"):
        """
        Reads a FASTA file and extracts gene and intergenic (IR) sequences. The method parameter controls how sequences are combined:

        :param filter_xs: Only gene sequences are used. No IR sequences are included.
        :param sort_by_genes: Sort the isolate sequences by gene names (alphanumeric).
        :param use_gene_file: Load all available genes in fasta file or use the gene_file.
        :param method: format of gene and intergenic sequences ["no_intergenic", "merged", "separate"].
        :return a list of gene/intergenic sequences within the isolate fasta file
        """

        if use_gene_file:
            sorted_genes = {key: sorted(value) for key, value in self.genes.items()}
            gene_list = [gene.lower() for gene in sorted_genes[target]]
            gene_sequences = {gene: {'sequence': None, 'before_ir': None, 'after_ir': None}
                              for gene in gene_list}
        else:
            gene_list = None
            sorted_genes = set()

            try:
                with open(file_path, 'r') as file:
                    for line in file:
                        if line.startswith('>'):
                            gene_name = line.split("|")[-3].strip("]").lower()
                            if "ir" not in gene_name:
                                sorted_genes.add(gene_name)
            except Exception as e:
                logging.error(f"Error reading file {file_path}: {e}")
                return [], []

            sorted_genes = sorted(sorted_genes)
            gene_sequences = {}
        file_gene_order = []
        temp_dict = {'before_ir': None, 'sequence': None, 'after_ir': None, 'gene_name': None}

        try:
            with open(file_path, 'r') as file:
                sequence = ''
                current_header = None

                def process_temp_dict():
                    """Helper function to process and clear temporary dictionary"""
                    if temp_dict['gene_name']:
                        gene_name = temp_dict['gene_name']
                        # Initialize gene_sequences entry if it doesn't exist
                        if gene_name not in gene_sequences:
                            gene_sequences[gene_name] = {'sequence': None, 'before_ir': None, 'after_ir': None}
                            if gene_name not in file_gene_order:
                                file_gene_order.append(gene_name)
                        if temp_dict['sequence']:
                            gene_sequences[gene_name]['sequence'] = temp_dict['sequence']
                        if temp_dict['before_ir']:
                            gene_sequences[gene_name]['before_ir'] = temp_dict['before_ir']
                        if temp_dict['after_ir']:
                            gene_sequences[gene_name]['after_ir'] = temp_dict['after_ir']

                    # Clear temporary dictionary
                    temp_dict['before_ir'] = None
                    temp_dict['sequence'] = None
                    temp_dict['after_ir'] = None
                    temp_dict['gene_name'] = None

                for line in file:
                    if "<<P>>" in line:
                        return None

                    line = line.strip()
                    if line.startswith('>'):
                        # Process previous sequence
                        if sequence:
                            clean_sequence = sequence.replace('\n', '').upper()

                            if current_header and 'IR:' in current_header:
                                if '|BEFORE|' in current_header:
                                    # If we already have a before_ir or sequence, process the previous entry
                                    if temp_dict['before_ir'] or temp_dict['sequence']:
                                        process_temp_dict()
                                    temp_dict['before_ir'] = clean_sequence
                                elif '|AFTER|' in current_header:
                                    temp_dict['after_ir'] = clean_sequence
                                    # After finding an AFTER IR, process the complete entry
                                    process_temp_dict()
                            else:
                                # This is a gene sequence
                                gene_name = current_header.split("|")[-3].strip("]").lower()
                                # Check if gene should be processed
                                if (use_gene_file and gene_name in gene_list) or not use_gene_file:
                                    # If we already have a sequence, process the previous entry
                                    if temp_dict['sequence']:
                                        process_temp_dict()
                                    temp_dict['sequence'] = clean_sequence
                                    temp_dict['gene_name'] = gene_name

                        sequence = ''
                        current_header = line
                    else:
                        sequence += line.upper()

                # Process the last sequence
                if sequence:
                    clean_sequence = sequence.replace('\n', '').upper()
                    if current_header and 'IR:' in current_header:
                        if '|BEFORE|' in current_header:
                            temp_dict['before_ir'] = clean_sequence
                        elif '|AFTER|' in current_header:
                            temp_dict['after_ir'] = clean_sequence
                    else:
                        gene_name = current_header.split("|")[-3].strip("]").lower()
                        if (use_gene_file and gene_name in gene_list) or not use_gene_file:
                            temp_dict['sequence'] = clean_sequence
                            temp_dict['gene_name'] = gene_name

                    # Process the final entry
                    process_temp_dict()

        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            return None

        # Choose which gene order to use
        if use_gene_file:
            genes_to_process = sorted_genes[target] if sort_by_genes else file_gene_order
        else:
            genes_to_process = sorted_genes

        combined_sequences = []
        genes_in_isolate = []
        missing = []

        for gene in genes_to_process:
            gene_lower = gene.lower()
            gene_data = gene_sequences.get(gene_lower)

            if gene_data:
                sequence = gene_data['sequence']
                ir_before = gene_data['before_ir']
                ir_after = gene_data['after_ir']

                # Check for missing gene
                is_missing = (sequence is None or "X" in sequence)

                if method == "no_intergenic":
                    # Only gene sequences
                    if is_missing:
                        combined_sequences.append("XXXXXXXXXXXXXXXXXX")
                    else:
                        combined_sequences.append(sequence)
                    genes_in_isolate.append(gene)

                elif method == "merged":
                    # Merge IR + gene + IR into a single sequence
                    # If IR is missing, use placeholder
                    if ir_before is None:
                        ir_before = "YYYYYYYYYYYY"
                    if ir_after is None:
                        ir_after = "YYYYYYYYYYYY"
                    if is_missing:
                        combined_sequences.append("XXXXXXXXXXXXXXXXXX")
                    else:
                        merged_seq = ir_before + sequence + ir_after
                        combined_sequences.append(merged_seq)
                    genes_in_isolate.append(gene)

                else:  # method == "separate"
                    # The original behavior: [IR_before, gene, IR_after]
                    if ir_before is None:
                        ir_before = "YYYYYYYYYYYY"
                    combined_sequences.append(ir_before)

                    if is_missing:
                        combined_sequences.append("XXXXXXXXXXXXXXXXXX")
                    else:
                        combined_sequences.append(sequence)

                    if ir_after is None:
                        ir_after = "YYYYYYYYYYYY"
                    combined_sequences.append(ir_after)

                    # Add all three entries to genes_in_isolate
                    genes_in_isolate.append(gene + "_ir_before")
                    genes_in_isolate.append(gene)
                    genes_in_isolate.append(gene + "_ir_after")

                if is_missing:
                    missing.append(gene)

            else:
                # Gene is missing
                missing.append(gene)
                if method == "no_intergenic":
                    # Just gene placeholder
                    combined_sequences.append("XXXXXXXXXXXXXXXXXX")
                    genes_in_isolate.append(gene)

                elif method == "merged":
                    # Merged but gene missing - just placeholder
                    combined_sequences.append("XXXXXXXXXXXXXXXXXX")
                    genes_in_isolate.append(gene)

                else:  # separate
                    # Add placeholders for IR and gene
                    combined_sequences.extend(["YYYYYYYYYYYY", "XXXXXXXXXXXXXXXXXX", "YYYYYYYYYYYY"])
                    genes_in_isolate.append(gene + "_ir_before")
                    genes_in_isolate.append(gene)
                    genes_in_isolate.append(gene + "_ir_after")

        # print(f"\nMISSING {len(missing)} out of {len(genes_to_process)}")
        # print(missing)

        if filter_xs and method == "separate":
            # For separate mode, filtering out genes with 'X' in them
            filtered_sequences = []
            filtered_genes = []

            # Since in 'separate' mode each gene block is 3 sequences (IR_before, gene, IR_after)
            for i in range(0, len(combined_sequences), 3):
                if "X" not in combined_sequences[i + 1]:  # Check gene sequence part
                    filtered_sequences.extend(combined_sequences[i:i + 3])
                    # (i//3)*3 in genes_in_isolate corresponds to IR_before in that block, so the gene is (i//3)*3+1
                    filtered_genes.extend(genes_in_isolate[i:i + 3])

            combined_sequences = filtered_sequences
            genes_in_isolate = filtered_genes
        elif filter_xs and method in ["no_intergenic", "merged"]:
            # In no_intergenic or merged mode, each element corresponds to one gene
            filtered_sequences = []
            filtered_genes = []
            for seq, g in zip(combined_sequences, genes_in_isolate):
                if "X" not in seq:
                    filtered_sequences.append(seq)
                    filtered_genes.append(g)
            combined_sequences = filtered_sequences
            genes_in_isolate = filtered_genes

        if not combined_sequences:
            logging.warning(f"No valid sequences found in file {file_path}")

        return combined_sequences, genes_in_isolate