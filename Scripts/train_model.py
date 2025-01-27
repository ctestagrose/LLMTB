import argparse
import json
import os
import glob
import random
from src.data_prep import DataPreparer
from src.Utils.Gene_Manager import GeneManager
from src.Utils.Sequence_Processor import SequenceProcessor
from src.Utils.Tokenizer_Manager import TokenizerManager
from src.Utils.Trainer import Trainer

def parse_arguments():
    """
    Parse command line arguments

    :return: parsed arguments
    """

    # Load JSON file that contains the parameters used for training
    with open('./Config/train_config.json', 'r') as f:
        file_args = json.load(f)

    # Initialize the argument parset
    parser = argparse.ArgumentParser(description='Train a BERT model for MTB Antibiotic Resistance.')

    # Define the arguments
    parser.add_argument('--sequence_dir', type=str, required=True, help='Path to the directory of fasta files.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--model_config', type=str, required=True, help='Path to model configuration')
    parser.add_argument('--antibiotic', type=str, required=True, help='All, Rare, multi-cat, multi-cat rare, multi-cat all, or provide one antibiotic abbrev.')
    parser.add_argument('--save_path', type=str, required=True, help='Where to save model?')
    parser.add_argument('--use_holdout', action='store_true', help='Use a hold out test set?')
    parser.add_argument('--use_gene_file', action='store_true', help='Use all the genes in a fasta or the gene file?')
    parser.add_argument('--gene_file', type=str, required=True, help='Path to the genes JSON file.')
    parser.add_argument('--oversample', action='store_true', help='Oversample data?')
    parser.add_argument('--adversarial_training', action='store_true', help='Include adversarial adjustment?')
    parser.add_argument('--target_file', type=str, required=True, help='Path to the targets file.')

    # Convert the file_args dictionary into "command-line" arguments
    argument_list = []
    for key, value in file_args.items():
        if isinstance(value, bool):
            if key == "use_holdout" and value:
                argument_list.append(f'--{key}')
            elif key == "use_gene_file" and value:
                argument_list.append(f'--{key}')
            elif key == "oversample" and value:
                argument_list.append(f'--{key}')
            elif key == "adversarial_training" and value:
                argument_list.append(f'--{key}')
        else:
            argument_list.append(f'--{key}')
            argument_list.append(str(value))

    # parse the "command-line" arguments
    args = parser.parse_args(argument_list)

    return args


if __name__ == '__main__':
    # Parse the arguments in the parameters files
    args = parse_arguments()

    # Define the list of antibiotics based on the selected option 
    if args.antibiotic.lower() == "all":
        antibiotics = ["AMI", "INH", "RIF", "LEV", "ETH", "EMB", "RFB", "MXF", "KAN", "LZD", "BDQ", "DLM", "CFZ"]
    elif args.antibiotic.lower() == "rare" or args.antibiotic.lower() == "rare_genes_drugs":
        antibiotics = ["LZD", "BDQ", "DLM", "CFZ"]
    elif args.antibiotic.lower() == "multi-cat":
        antibiotics = ["multi-cat"]
    elif args.antibiotic.lower() == "multi-cat rare":
        antibiotics = ["multi-cat rare"]
    elif args.antibiotic.lower() == "multi-cat all":
        antibiotics = ["multi-cat all"]
    else:
        antibiotics = [args.antibiotic]


    # Train model for each antibiotic
    for antibiotic in antibiotics:
        args.antibiotic = antibiotic

        # Create the folder where model files will be saved
        save_path = os.path.join(args.save_path, args.antibiotic)
        os.makedirs(save_path, exist_ok=True)
        
        # Glob the fasta files for specific antibiotic 
        fasta_files = glob.glob(os.path.join(args.sequence_dir, '**', '*.fasta'), recursive=True)

        random.Random(17).shuffle(fasta_files)
        
        # Initialize gene_manager, sequence_processor, tokenizer_manager, and data_prep
        gene_manager = GeneManager(args.gene_file)
        sequence_processor = SequenceProcessor()
        tokenizer_manager = TokenizerManager()
        
        data_preparer = DataPreparer(gene_manager, sequence_processor, tokenizer_manager, args)

        zipped_data = data_preparer.prep_data(
            fasta_files=fasta_files,
            target_file=args.target_file,
            target_format=args.antibiotic,
            mode="Train"
        )

        trainer = Trainer(args, sequence_processor, tokenizer_manager, gene_manager)
        trainer.train(zipped_data, mode="Train")
