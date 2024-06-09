import os
import tqdm
import torch
import tarfile
import random
import argparse
import numpy as np
import pandas as pd
from datetime import date
from utilities.data import MaterialData
from models.crystal_model import ProDosNet
from utilities.preprocess import CrystalGraphPDOS
from torch_geometric.loader import DataLoader
from utilities.training import run_cross_validation, run_test

def main(args):
    # Set random seeds 
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    args.cuda = torch.cuda.is_available()
    if args.save_path is None:
        save_path = 'test_outputs/'+str(date.today())+'_'+args.name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = args.save_path
        assert os.path.exists(save_path), print("Please, provide a valid save_path")

    # If task is preprocess, run data preprocessing 
    if args.task == "preprocess":
        assert args.preprocess_ids is not None, "Please provide a csv file with list of material ids for preprocessing."
        assert os.path.exists(args.preprocess_ids), 'Provided csv file with list of material ids for preprocessing does not exist!'
        assert args.cif_dir is not None, "Please provide directory with material cif files."
        assert os.path.exists(args.cif_dir), 'Provided directory with material cif files does not exist!'
        assert args.dos_dir is not None, "Please provide directory with material PDOS data."
        assert os.path.exists(args.dos_dir), 'Provided directory with material PDOS data does not exist!'

        # Create directory where processed data will be saved
        if args.save_data_dir is None:
            if not os.path.exists('processed_data'):
                os.makedirs('processed_data')
            args.save_data_dir = 'processed_data'
        else:
            assert os.path.exists(args.save_data_dir), "Provided save_data_dir path does not exist"
        
        if not os.path.exists(f'{args.save_data_dir}/processed_data'):
            os.makedirs(f'{args.save_data_dir}/processed_data')

        preprocess_ids = pd.read_csv(args.preprocess_ids)

        graph_gen = CrystalGraphPDOS(dos_dir=args.dos_dir,
                                     cif_dir=args.cif_dir, 
                                     max_num_nbr=args.max_num_nbr, 
                                     max_element=args.max_element,
                                     radius=args.radius,
                                     sigma=args.sigma, 
                                     grid=args.grid,
                                     norm_pdos = args.norm_pdos)
                                     
        failed_ids = []
        successful_ids = []
        for index, _ in tqdm.tqdm(preprocess_ids.iterrows(), total=preprocess_ids.shape[0]):
            cif_id = preprocess_ids.iloc[index, 0]
            graph = graph_gen.get_crystal_pdos_graph(os.path.join(args.cif_dir, f"{cif_id}.cif"))
            if graph is not None:
                torch.save(graph, os.path.join(args.save_data_dir, f'processed_data/{cif_id}_crystal_graph_pdos.pt'))
                successful_ids.append(cif_id)
            else:
                failed_ids.append(cif_id)

        print("-------------- Compressing processed data into tar.gz file . . . --------------- \n")
        with tarfile.open(f"{args.save_data_dir}/processed_data.tar.gz", "w:gz") as tar:
            tar.add(f"{args.save_data_dir}/processed_data", arcname=os.path.basename(f"{args.save_data_dir}/processed_data"))
        print(" Done compressing! \n ")

        n_failed = len(failed_ids)
        n_successful = len(successful_ids)

        failed_to_process_ids_file = os.path.join(args.save_data_dir, "failed_to_process_ids.csv")
        successfully_processed_ids_file = os.path.join(args.save_data_dir, "processed_ids.csv")

        successful_ids_df = pd.DataFrame({"processed_ids": successful_ids})
        successful_ids_df.to_csv(successfully_processed_ids_file, index=False)
        if failed_ids:
            failed_ids_df = pd.DataFrame({"failed_ids": failed_ids})
            failed_ids_df.to_csv(failed_to_process_ids_file, index=False)

        print("--------------------------- Finished processing data --------------------------- \n")
        print(f" Successfully processed {n_successful} materials. \n The list of processed ids can be found at {successfully_processed_ids_file}")
        print(f" Failed to process {n_failed} materials.") 
        if failed_ids:
            print(f" The list of failed ids can be found at {failed_to_process_ids_file}")
        print(f" Processed data saved to {args.save_data_dir}")


    # If task is cross_val, start cross-validation run
    if args.task == "cross_val":
        assert args.train_ids is not None, "Please provide list of material ids for cross-validation (train_ids.csv)."
        assert args.data_file is not None, "Please provide tar dataset file containing processed data."
        config = {"n_conv": args.n_conv, "weight_decay": args.wd}
        run_cross_validation(config, args, save_path)
       
    
    # If task is test, use pretrained model to predict PDOS on test set 
    if args.task == "test":
        config = {"n_conv": args.n_conv, "weight_decay": args.wd}
        assert args.test_ids is not None, "Please provide list of material ids for the test (test_ids.csv)."
        assert args.data_file is not None, "Please provide tar dataset file containing processed data."
        assert args.model is not None, "Please provide path to pretrained model"
        test_dataset = MaterialData(data_file=args.data_file, id_file=args.test_ids)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        test_ids_file_name = os.path.basename(args.test_ids)
        
        model = ProDosNet(orig_atom_fea_len=test_dataset[0].x.shape[1], nbr_fea_len=test_dataset[0].edge_attr.shape[1], n_conv=config["n_conv"], use_mlp=args.use_mlp)

        if args.cuda:
            device = torch.device("cuda")
            model.to(device)
        print(f"------------ Running Test on {test_ids_file_name} ------------- \n")
        run_test(args, save_path, test_loader, model)
        print()
        print("------------------------- Finished Test ------------------------ \n")


    # If task is predict, use pretrained model to predict on data that does not have target values for PDOS 
    if args.task == "predict":
        assert args.predict_ids is not None, "Please provide list of material ids to preprocess."
        assert args.cif_dir is not None, "Please provide directory with material cif files." 

        if args.save_data_dir is None:
            if not os.path.exists('processed_data'):
                os.makedirs('processed_data')
            args.save_data_dir = 'processed_data'

        preprocess_data = ""

        graph_gen = CrystalGraphPDOS(dos_dir=args.dos_dir,
                                     cif_dir=args.cif_dir, 
                                     max_num_nbr=args.max_num_nbr, 
                                     max_element=args.max_element,
                                     bound_high=args.bound_high,
                                     bound_low=args.bound_low,  
                                     radius=args.radius,
                                     sigma=args.sigma, 
                                     grid=args.grid)
                                     
        for index, _ in tqdm(preprocess_data.iterrows()):
            cif_id = preprocess_data.iloc[index, 0]
            graph = graph_gen.get_crystal_pdos_graph_pred(os.path.join(args.cif_dir, f"{cif_id}.cif"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PDOS Net")

    parser.add_argument("--data_dir", default=None,
                        help="Provide data directory. Default: None")

    parser.add_argument("--data_file", default=None,
                        help="Provide dataset tar file name. Default: None")
    
    parser.add_argument("--save_path", default=None,
                        help="Provide path where output will be saved. Default: None")

    parser.add_argument("--name", default="ProDosNet_experiment",
                        help="Provide experiment name. Default: ProDosNet_experiment")

    parser.add_argument("--model_name", default="crystal_model",
                        help="Provide model name. Default: crystal_model")

    parser.add_argument("--task", choices=['preprocess', 'cross_val', 'test', 'predict'], default="train_cv",
                        help="Choose task from available options ")
                        
    parser.add_argument("--train_ids", default="train_ids.csv",
                        help="Provide csv file with mp-ids for k-fold cross-validation. Default: train_ids.csv")

    parser.add_argument("--test_ids", default=None,
                        help="Provide csv file with mp-ids for testing. Default: None")

    parser.add_argument("--predict_ids", default=None,
                        help="Provide csv file with mp-ids for prediction. Default: None")
    
    parser.add_argument("--preprocess_ids", default=None,
                        help="Provide csv file with mp-ids for preprocessing. Default: None")

    parser.add_argument("--cif_dir", default=None,
                        help="Provide directory with cif files. Default: None")

    parser.add_argument("--dos_dir", default=None,
                        help="Provide directory with PDOS files. Default: None")
    
    parser.add_argument("--save_data_dir", default=None,
                        help="Provide directory where to save preprocessed data. Default: None")
    
    parser.add_argument("--max_num_nbr", default=12, type=int, 
                        help="Provide maximum number of neighbours in crystal graph. Default: 12")

    parser.add_argument("--max_element", default=83, type=int, 
                        help="Provide atomic number of most heavy element allowed in dataset. Default: 83")
    
    parser.add_argument("--radius", default=8.0, type=float, 
                        help="Provide maximum distance to neighbour in crystal graph in A. Default: 8 A")
    
    parser.add_argument("--sigma", default=0.3, type=float, 
                        help="Provide DOS broadening parameter. Default: 0.2")
    
    parser.add_argument("--grid", default=256, type=int, 
                        help="Provide number of grid points to represent DOS. Default: 256")
    
    parser.add_argument("--model_config", default="utilites/default_config.json",
                        help="Provide model configuration. Default: default_config.json")

    parser.add_argument("--model", default=None,
                        help="Provide pretrained model. Default: None")

    parser.add_argument("--optim", choices=['Adam', 'AdamW'], default="Adam",
                        help="Select optimizer. Default: Adam")

    parser.add_argument("--batch_size", default=32, type=int, 
                        help="Provide batch size. Default: 32")

    parser.add_argument("--lr", default=0.001, type=float, 
                        help="Provide learning rate value. Default: 0.001")

    parser.add_argument("--n_conv", default=3, type=int,
                        help="Provide number of graph convolution layers. Default: 2")
    
    parser.add_argument("--wd", default=0.0, type=float,
                        help="Provide weight decay value. Default: 0.0")

    parser.add_argument("--epochs", default=100, type=int,
                        help="Provide number of epochs for training the model. Default: 100")

    parser.add_argument("--kfold", default=5, type=int,
                        help="Provide number of folds for k-fold cross validation. Default: 5")

    parser.add_argument("--skip_embedding", default=False, action='store_true',
                        help="Skip embedding layes in model. Default: False")

    parser.add_argument("--train_on_dos", default=False, action='store_true',
                        help="If True, trains on total DOS instead of orbital PDOS. Default: False")

    parser.add_argument("--train_on_atomic_dos", default=False, action='store_true',
                        help="If True, trains on atomic DOS instead of orbital PDOS. Default: False")

    parser.add_argument("--plot_training", default=False, action='store_true',
                        help="If True, plots training curve and saves it in output directory. Default: False")

    parser.add_argument("--plot_interval", default=100, type=int,
                        help="Saves training curve each *plot_interval* epochs. If None, saves training curve only after last epoch. Default: 100")

    parser.add_argument("--model_save_interval", default=100, type=int,
                        help="Saves model each *model_save_interval* epochs. Default: 100")
                        
    parser.add_argument("--save_best_model", default=True, type=bool,
                        help="If True, saves best model for each fold of cross-validation. Default: True")

    parser.add_argument("--save_pdos", default=False, action='store_true',
                        help="Saves predicted orbital PDOS for cross-validation run. Default: False")

    parser.add_argument("--save_dos", default=False, action='store_true',
                        help="Saves predicted DOS for cross-validation run. Default: False")
    
    parser.add_argument("--save_only_1_fold_val_pred", default=True, action='store_false',
                        help="Saves predicted PDOS only for first fold during cross-validation run. Default: True")
                        
    parser.add_argument("--use_cdf", default=False, action='store_true',
                        help="If True, uses RMSE of CDFs as loss function. Default False")
                        
    parser.add_argument("--use_mlp", default=False, action='store_true',
                        help="If True, uses fully connected NN in graph convolution. Default: False")
    
    parser.add_argument("--scale", default=False, action='store_true',
                        help="If True, rescales bond distances to [0, 1] range. Default: False")
    
    parser.add_argument("--norm_pdos", default=True, action='store_false',
                        help="If True, normalize orbital PDOS to area = 1. Default: True")

    args = parser.parse_args()
    
    main(args)
