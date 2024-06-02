import torch
import itertools
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from utilities.utils import Scaler
from utilities.data import MaterialData
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from models.crystal_model import ProDosNet
from torch.nn.functional import mse_loss
from utilities.utils import save_model, save_training_curves, save_cv_results, print_output, plot_training_curve
#import matplotlib.pyplot as plt

def run_cross_validation(config: dict, args, save_path: str):
    """
        Run k-fold cross-validation on training/validation dataset
        ----------------------------------------------------------
        Input:
            - config:             Hyperparameters for this experiment
            - args:               Set of user-defined arguments
            - save_path:          Path where outputs will be saved
        Output:
            - Saves cross-validation results in csv file
            - Saves and plots trainig curves
            - Saves models checkpoints 
    """
    if args.model_name == "crystal_model":
        from models.crystal_model import ProDosNet
    else:
        print(f"Model {args.model_name} is not available")
        return None

    cv_results_list = []

    print("------------------- Starting Cross-Validation ------------------ \n")
    print(" Running {}-fold cross-validation on {} data file \n".format(args.kfold, args.train_ids))
    dataset = MaterialData(data_file=args.data_file, id_file=args.train_ids)
    splits = KFold(n_splits=args.kfold, shuffle=True, random_state=42)

    print("------------------------ Training Model ------------------------ \n")
    for fold, (train_ids, val_ids) in enumerate(splits.split(np.arange(len(dataset)))):

        model = ProDosNet(orig_atom_fea_len=dataset[0].x.shape[1], nbr_fea_len=dataset[0].edge_attr.shape[1], n_conv=config["n_conv"], use_mlp=args.use_mlp, use_cdf=args.use_cdf)

        if args.cuda:
            device = torch.device("cuda")
            model.to(device)

        metric = nn.MSELoss()
        if args.optim == "Adam":
            print(" Using Adam optimizer")
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        elif args.optim == "AdamW":
            print(" Using AdamW optimizer")
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

        print("---------------------------- Fold {} ----------------------------".format(fold+1))
        

        fold_training_curves_dict = {'train_loss': [],
                                     'val_loss': [],
                                     'train_dos_mse': [],
                                     'val_dos_mse': [],
                                     'train_dos_mse_cdf': [],
                                     'val_dos_mse_cdf': [],
                                     'train_atomic_dos_mse': [],
                                     'val_atomic_dos_mse': [],
                                     'train_atomic_dos_mse_cdf': [],
                                     'val_atomic_dos_mse_cdf': [],
                                     'train_orbital_pdos_mse': [],
                                     'val_orbital_pdos_mse': [],
                                     'train_orbital_pdos_mse_cdf': [],
                                     'val_orbital_pdos_mse_cdf': [],
                                     'model_weight_sum': []}
        
        fold_results_df = pd.DataFrame({'error': list(fold_training_curves_dict.keys())})

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        validation_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_subsampler)
        validation_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=validation_subsampler)

        # Save cross-validation id splits
        train_mp_ids = []
        training_distances = []
        for data in train_loader:
            training_distances.append(torch.flatten(data.edge_attr))
            train_mp_ids.extend(data.material_id)
        train_mp_ids_df = pd.DataFrame({"train_ids": train_mp_ids})
        train_mp_ids_df.to_csv(f'{save_path}/train_ids_fold_{fold+1}.csv', index=False)

        val_mp_ids = []
        for data in validation_loader:
            val_mp_ids.extend(data.material_id)
        val_mp_ids_df = pd.DataFrame({"val_ids": val_mp_ids})
        val_mp_ids_df.to_csv(f'{save_path}/val_ids_fold_{fold+1}.csv', index=False)

        # Create a Scaler for bond distances 
        if args.scale:
            training_distances = torch.cat(training_distances, dim=0)
            scaler = Scaler(training_distances)
        else:
            scaler = None
        
        # Train the model
        for epoch in range(1, args.epochs+1):
            train_error_dict = train(model, 
                                     optimizer, 
                                     metric, 
                                     train_loader, 
                                     train_on_dos=args.train_on_dos, 
                                     train_on_atomic_dos=args.train_on_atomic_dos,
                                     use_cuda=args.cuda, 
                                     use_cdf=args.use_cdf, 
                                     scaler=scaler,
                                     epoch=epoch)
            
            val_error_dict = validation(model, 
                                        metric, 
                                        epoch, 
                                        fold, 
                                        save_path, 
                                        validation_loader, 
                                        train_on_dos=args.train_on_dos, 
                                        train_on_atomic_dos=args.train_on_atomic_dos,
                                        use_cuda=args.cuda, 
                                        use_cdf=args.use_cdf, 
                                        scaler=scaler)
            
            for key, item in train_error_dict.items():
                fold_training_curves_dict[key].append(item)

            for key, item in val_error_dict.items():
                fold_training_curves_dict[key].append(item)

            if epoch%args.model_save_interval == 0:
                train_error_ws_dict = train_error_dict
                train_error_ws_dict['model_weight_sum'] = total_weight_sum
                error_df = pd.concat([pd.DataFrame.from_dict(train_error_ws_dict, orient='index', columns=[f'epoch_{epoch}']),
                                      pd.DataFrame.from_dict(val_error_dict, orient='index', columns=[f'epoch_{epoch}'])])
                error_df['error'] = error_df.index
                
                min_error_list = [np.min(error_list) for _, error_list in fold_training_curves_dict.items()]
                fold_results_df[f'min_error'] = min_error_list
                fold_results_df[f'epoch_{epoch}'] = fold_results_df.merge(error_df, on='error', how='inner')[f'epoch_{epoch}']

                fold_results_df.to_csv(f'{save_path}/results_fold_{fold+1}.csv', index=False)
            
            print_output(epoch, 
                         train_error_dict['train_loss'], 
                         val_error_dict['val_loss'], 
                         train_error_dict['train_orbital_pdos_mse'], 
                         val_error_dict['val_orbital_pdos_mse'], 
                         train_error_dict['train_orbital_pdos_mse_cdf'], 
                         val_error_dict['val_orbital_pdos_mse_cdf'])
            
            val_loss = val_error_dict['val_loss']
            train_loss = train_error_dict['train_loss']
            val_pdos_rmse = val_error_dict['val_orbital_pdos_mse']
            train_pdos_rmse = train_error_dict['train_orbital_pdos_mse']
            val_cdf_pdos_rmse = val_error_dict['val_orbital_pdos_mse_cdf']
            train_cdf_pdos_rmse = train_error_dict['train_orbital_pdos_mse_cdf']

            n_parameters = 0
            layer_sum_list = []
            for parameters in model.parameters():
                n_parameters += torch.numel(parameters)
                parameters_np = parameters.cpu().data.numpy()
                layer_sum_list.append(np.sum(np.abs(parameters_np)))
            total_weight_sum = np.sum(layer_sum_list).item()/n_parameters

            fold_training_curves_dict['model_weight_sum'].append(total_weight_sum)
       
            if epoch == 1:
                best_val_loss = val_loss
                best_train_loss = train_loss
                best_val_pdos_rmse = val_pdos_rmse
                best_train_pdos_rmse = train_pdos_rmse
                best_val_cdf_pdos_rmse = val_cdf_pdos_rmse
                best_train_cdf_pdos_rmse = train_cdf_pdos_rmse
         
                model_state_best = {'epoch': epoch, 'state_dict': model.state_dict(), 'best_val_loss': best_val_loss, 'best_train_loss': best_train_loss,
                                    'best_val_pdos_rmse': best_val_pdos_rmse, 'best_train_pdos_rmse': best_train_pdos_rmse, 'best_val_cdf_pdos_rmse': best_val_cdf_pdos_rmse, 'best_train_cdf_pdos_rmse': best_train_cdf_pdos_rmse, 'optimizer': optimizer.state_dict(), 'args': vars(args)}
            else:
                best = val_loss < best_val_loss
                
                if best:
                    best_val_loss = val_loss
                    best_train_loss = train_loss
                    best_val_pdos_rmse = val_pdos_rmse
                    best_train_pdos_rmse = train_pdos_rmse
                    best_val_cdf_pdos_rmse = val_cdf_pdos_rmse
                    best_train_cdf_pdos_rmse = train_cdf_pdos_rmse
                    

                    model_state_best = {'epoch': epoch, 'state_dict': model.state_dict(), 'best_val_loss': best_val_loss, 'best_train_loss': best_train_loss,
                                    'best_val_pdos_rmse': best_val_pdos_rmse, 'best_train_pdos_rmse': best_train_pdos_rmse, 'best_val_cdf_pdos_rmse': best_val_cdf_pdos_rmse, 'best_train_cdf_pdos_rmse': best_train_cdf_pdos_rmse, 'optimizer': optimizer.state_dict(), 'args': vars(args)}
            
                    if args.save_best_model:
                        save_model(model_state_best, epoch, save_path, fold=fold, best=True)

            if args.plot_training and epoch % args.plot_interval == 0:
                plot_training_curve(save_path, val_loss_list=fold_training_curves_dict['val_loss'], train_loss_list=fold_training_curves_dict['train_loss'], fold=fold+1)

            if epoch % args.model_save_interval==0:
                model_state = {'epoch': epoch, 'state_dict': model.state_dict(), 'best_val_loss': best_val_loss, 'best_train_loss': best_train_loss,
                                    'best_val_pdos_rmse': best_val_pdos_rmse, 'best_train_pdos_rmse': best_train_pdos_rmse, 'best_val_cdf_pdos_rmse': best_val_cdf_pdos_rmse, 'best_train_cdf_pdos_rmse': best_train_cdf_pdos_rmse, 'optimizer': optimizer.state_dict(), 'args': vars(args)}
                save_model(model_state, epoch, save_path, fold)

                save_training_curves(fold+1, fold_training_curves_dict, save_path)

        

        
        save_training_curves(fold+1, fold_training_curves_dict, save_path)


    print("----------------- Finished Cross-Validation -----------------")
   
    #save_cv_results(args.kfold, training_curve_name_list, cv_lists, mean_list, std_list, save_path)

    if args.save_dos or args.save_pdos:
        print("------------------- Saving Predicted PDOS -------------------")
        print()
        for fold, (train_ids, val_ids) in enumerate(splits.split(np.arange(len(dataset)))):
            print("--------------------------- Fold {} --------------------------".format(fold+1))
            validation_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
            validation_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=validation_subsampler)
            best_model = torch.load('{}/checkpoint_fold_{}_best.pth.tar'.format(save_path, fold+1), map_location=torch.device('cpu'))
            model.load_state_dict(best_model['state_dict'])
            optimizer.load_state_dict(best_model['optimizer'])
            if args.cuda:
                device = torch.device("cuda")
                model.to(device)
            val_error_dict = validation(model, 
                                        metric, 
                                        "best", 
                                        fold, 
                                        save_path,
                                        validation_loader, 
                                        train_on_dos=args.train_on_dos, 
                                        train_on_atomic_dos=args.train_on_atomic_dos, 
                                        save_output=True, 
                                        save_dos=args.save_dos, 
                                        save_pdos=args.save_pdos, 
                                        use_cuda=args.cuda, 
                                        use_cdf=args.use_cdf, 
                                        scaler=scaler)
            if args.save_only_1_fold_val_pred:
                break
        print("---------------- Finished Saving Predictions ----------------")
          


def run_test(args, save_path: str, test_loader: DataLoader, model: ProDosNet):
    metric = nn.MSELoss()
    pretrained_model = torch.load(args.model, map_location=torch.device('cpu'))
    model.load_state_dict(pretrained_model['state_dict'])
    if args.cuda:
            device = torch.device("cuda")
            model.to(device)
    error_dict = validation(
                model, metric, epoch=0, fold=None, save_path=save_path, validation_loader=test_loader, train_on_dos=args.train_on_dos, save_output=True, save_dos=args.save_dos, save_pdos=args.save_pdos, use_cuda=args.cuda, use_cdf=args.use_cdf, train_on_atomic_dos=args.train_on_atomic_dos, test=True)
    result_df = pd.DataFrame(data=error_dict.items())
    result_df.to_csv("%s/results.csv"%save_path, sep='\t')

    print("------------------------- Test Results ------------------------- \n")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(" \n Total training results: ")
        print(result_df.to_string(index=False))



def train(model: ProDosNet = None,
          optimizer: torch.optim = None,
          metric: nn.MSELoss = None, 
          train_loader: DataLoader = None, 
          use_cuda: bool = False, 
          use_cdf: bool = False, 
          train_on_dos: bool = False,
          train_on_atomic_dos: bool = False,
          scaler: Scaler = None,
          epoch: int = None) -> dict[str, float]:
    
    model.train()
    
    n_iter = len(train_loader)
    running_loss = 0.0
    running_dos_mse = 0.0
    running_dos_mse_cdf = 0.0
    running_atomic_dos_mse = 0.0
    running_atomic_dos_mse_cdf = 0.0
    running_orbital_pdos_mse = 0.0
    running_orbital_pdos_mse_cdf = 0.0

    for batch_idx, data in enumerate(train_loader):
        e_diff = torch.mean(data.e_diff)
    
        if use_cuda:
            device = torch.device('cuda')
            data = data.to(device)

        if train_on_dos:
            if use_cdf: 
                if scaler is not None:
                    edge_attr = scaler.norm(data.edge_attr)
                else: 
                    edge_attr = data.edge_attr
                
                target_dos = data.dos_cdf
                target_atomic_dos = data.atomic_dos_cdf
                target_orbital_pdos = data.pdos_cdf
         
                output_pdos, output_atomic_dos, output_dos = model(data.x, data.edge_index, edge_attr, data.batch, data.atoms_batch, use_cdf=use_cdf, train_on_pdos=False)

                loss = metric(output_dos/torch.max(target_dos, dim=1)[0].unsqueeze(1), target_dos/torch.max(target_dos, dim=1)[0].unsqueeze(1))
                loss_item = loss.item()
                
                output_dos_diff = torch.diff(output_dos, dim=1)/e_diff
                output_dos_diff[output_dos_diff<0] = 0.0
                dos_mse = mse_loss(output_dos_diff, (torch.diff(target_dos, dim=1)/e_diff)).item()
                dos_mse_cdf = mse_loss(output_dos/torch.max(target_dos, dim=1)[0].unsqueeze(1), target_dos/torch.max(target_dos, dim=1)[0].unsqueeze(1)).item()

                output_atomic_dos_diff = torch.diff(output_atomic_dos, dim=1)/e_diff
                output_atomic_dos_diff[output_atomic_dos_diff<0] = 0.0
                atomic_dos_mse = mse_loss(output_atomic_dos_diff, (torch.diff(target_atomic_dos, dim=1)/e_diff)).item()
                atomic_dos_mse_cdf = mse_loss(output_atomic_dos, target_atomic_dos).item()

                output_orbital_pdos_diff = torch.diff(output_pdos, dim=1)/e_diff
                output_orbital_pdos_diff[output_orbital_pdos_diff<0] = 0.0
                orbital_pdos_mse = mse_loss(output_orbital_pdos_diff, (torch.diff(target_orbital_pdos, dim=1)/e_diff)).item()
                orbital_pdos_mse_cdf = mse_loss(output_pdos, target_orbital_pdos).item()

            else:
                if scaler is not None:
                    edge_attr = scaler.norm(data.edge_attr)
                else: 
                    edge_attr = data.edge_attr

                target_dos = data.dos
                target_atomic_dos = data.atomic_dos
                target_orbital_pdos = data.pdos
         
                output_pdos, output_atomic_dos, output_dos = model(data.x, data.edge_index, edge_attr, data.batch, data.atoms_batch, use_cdf=use_cdf, train_on_pdos=False)

                loss = metric(output_dos/torch.max(target_dos, dim=1)[0].unsqueeze(1), target_dos/torch.max(target_dos, dim=1)[0].unsqueeze(1))
                loss_item = loss.item()
            
                dos_mse = mse_loss(output_dos, target_dos).item()

                output_dos_cdf = torch.cumsum(output_dos, dim=1)*e_diff
                target_dos_cdf = torch.cumsum(target_dos, dim=1)*e_diff
                dos_mse_cdf = mse_loss(output_dos_cdf/torch.max(target_dos_cdf, dim=1)[0].unsqueeze(1), target_dos_cdf/torch.max(target_dos_cdf, dim=1)[0].unsqueeze(1)).item()

                atomic_dos_mse = mse_loss(output_atomic_dos, target_atomic_dos).item()
                atomic_dos_mse_cdf = mse_loss(torch.cumsum(output_atomic_dos, dim=1)*e_diff, torch.cumsum(target_atomic_dos, dim=1)*e_diff).item()

                orbital_pdos_mse = mse_loss(output_pdos, target_orbital_pdos).item()
                orbital_pdos_mse_cdf = mse_loss(torch.cumsum(output_pdos, dim=1)*e_diff, torch.cumsum(target_orbital_pdos, dim=1)*e_diff).item()

        elif train_on_atomic_dos:
            if use_cdf: 
                if scaler is not None:
                    edge_attr = scaler.norm(data.edge_attr)
                else: 
                    edge_attr = data.edge_attr
                
                target_dos = data.dos_cdf
                target_atomic_dos = data.atomic_dos_cdf
                target_orbital_pdos = data.pdos_cdf
         
                output_pdos, output_atomic_dos, output_dos = model(data.x, data.edge_index, edge_attr, data.batch, data.atoms_batch, use_cdf=use_cdf, train_on_pdos=False)

                loss = metric(target_atomic_dos, output_atomic_dos)
                loss_item = loss.item()
                
                output_dos_diff = torch.diff(output_dos, dim=1)/e_diff
                output_dos_diff[output_dos_diff<0] = 0.0
                dos_mse = mse_loss(output_dos_diff, (torch.diff(target_dos, dim=1)/e_diff)).item()
                dos_mse_cdf = mse_loss(output_dos, target_dos).item()

                output_atomic_dos_diff = torch.diff(output_atomic_dos, dim=1)/e_diff
                output_atomic_dos_diff[output_atomic_dos_diff<0] = 0.0
                atomic_dos_mse = mse_loss(output_atomic_dos_diff, (torch.diff(target_atomic_dos, dim=1)/e_diff)).item()
                atomic_dos_mse_cdf = mse_loss(output_atomic_dos, target_atomic_dos).item()

                output_orbital_pdos_diff = torch.diff(output_pdos, dim=1)/e_diff
                output_orbital_pdos_diff[output_orbital_pdos_diff<0] = 0.0
                orbital_pdos_mse = mse_loss(output_orbital_pdos_diff, (torch.diff(target_orbital_pdos, dim=1)/e_diff)).item()
                orbital_pdos_mse_cdf = mse_loss(output_pdos, target_orbital_pdos).item()

            else:
                if scaler is not None:
                    edge_attr = scaler.norm(data.edge_attr)
                else: 
                    edge_attr = data.edge_attr

                target_dos = data.dos
                target_atomic_dos = data.atomic_dos
                target_orbital_pdos = data.pdos
         
                output_pdos, output_atomic_dos, output_dos = model(data.x, data.edge_index, edge_attr, data.batch, data.atoms_batch, use_cdf=use_cdf, train_on_pdos=False)

                loss = metric(target_atomic_dos, output_atomic_dos)
                loss_item = loss.item()
            
                dos_mse = mse_loss(output_dos, target_dos).item()
                dos_mse_cdf = mse_loss(torch.cumsum(output_dos, dim=1)*e_diff, torch.cumsum(target_dos, dim=1)*e_diff).item()

                atomic_dos_mse = mse_loss(output_atomic_dos, target_atomic_dos).item()
                atomic_dos_mse_cdf = mse_loss(torch.cumsum(output_atomic_dos, dim=1)*e_diff, torch.cumsum(target_atomic_dos, dim=1)*e_diff).item()

                orbital_pdos_mse = mse_loss(output_pdos, target_orbital_pdos).item()
                orbital_pdos_mse_cdf = mse_loss(torch.cumsum(output_pdos, dim=1)*e_diff, torch.cumsum(target_orbital_pdos, dim=1)*e_diff).item()

        else:
            if use_cdf: 
                if scaler is not None:
                    edge_attr = scaler.norm(data.edge_attr)
                else: 
                    edge_attr = data.edge_attr
                
                target_dos = data.dos_cdf
                target_atomic_dos = data.atomic_dos_cdf
                target_orbital_pdos = data.pdos_cdf
         
                output_pdos, output_atomic_dos, output_dos = model(data.x, data.edge_index, edge_attr, data.batch, data.atoms_batch, use_cdf=use_cdf, train_on_pdos=True)

                loss = metric(target_orbital_pdos, output_pdos)
                loss_item = loss.item()
                
                output_dos_diff = torch.diff(output_dos, dim=1)/e_diff
                output_dos_diff[output_dos_diff<0] = 0.0
                dos_mse = mse_loss(output_dos_diff, (torch.diff(target_dos, dim=1)/e_diff)).item()
                dos_mse_cdf = mse_loss(output_dos, target_dos).item()

                output_atomic_dos_diff = torch.diff(output_atomic_dos, dim=1)/e_diff
                output_atomic_dos_diff[output_atomic_dos_diff<0] = 0.0
                atomic_dos_mse = mse_loss(output_atomic_dos_diff, (torch.diff(target_atomic_dos, dim=1)/e_diff)).item()
                atomic_dos_mse_cdf = mse_loss(output_atomic_dos, target_atomic_dos).item()

                output_orbital_pdos_diff = torch.diff(output_pdos, dim=1)/e_diff
                output_orbital_pdos_diff[output_orbital_pdos_diff<0] = 0.0
                orbital_pdos_mse = mse_loss(output_orbital_pdos_diff, (torch.diff(target_orbital_pdos, dim=1)/e_diff)).item()
                orbital_pdos_mse_cdf = mse_loss(output_pdos, target_orbital_pdos).item()

            else:
                if scaler is not None:
                    edge_attr = scaler.norm(data.edge_attr)
                else: 
                    edge_attr = data.edge_attr

                target_dos = data.dos
                target_atomic_dos = data.atomic_dos
                target_orbital_pdos = data.pdos
         
                output_pdos, output_atomic_dos, output_dos = model(data.x, data.edge_index, edge_attr, data.batch, data.atoms_batch, use_cdf=use_cdf, train_on_pdos=True)

                loss = metric(target_orbital_pdos, output_pdos)
                loss_item = loss.item()
            
                dos_mse = mse_loss(output_dos, target_dos).item()
                dos_mse_cdf = mse_loss(torch.cumsum(output_dos, dim=1)*e_diff, torch.cumsum(target_dos, dim=1)*e_diff).item()

                atomic_dos_mse = mse_loss(output_atomic_dos, target_atomic_dos).item()
                atomic_dos_mse_cdf = mse_loss(torch.cumsum(output_atomic_dos, dim=1)*e_diff, torch.cumsum(target_atomic_dos, dim=1)*e_diff).item()

                orbital_pdos_mse = mse_loss(output_pdos, target_orbital_pdos).item()
                orbital_pdos_mse_cdf = mse_loss(torch.cumsum(output_pdos, dim=1)*e_diff, torch.cumsum(target_orbital_pdos, dim=1)*e_diff).item()

        running_loss += loss_item

        running_dos_mse += dos_mse
        running_dos_mse_cdf += dos_mse_cdf

        running_atomic_dos_mse += atomic_dos_mse
        running_atomic_dos_mse_cdf += atomic_dos_mse_cdf

        running_orbital_pdos_mse += orbital_pdos_mse
        running_orbital_pdos_mse_cdf += orbital_pdos_mse_cdf

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss/n_iter
    epoch_dos_mse = running_dos_mse/n_iter
    epoch_dos_mse_cdf = running_dos_mse_cdf/n_iter

    epoch_atomic_dos_mse = running_atomic_dos_mse/n_iter
    epoch_atomic_dos_mse_cdf = running_atomic_dos_mse_cdf/n_iter

    epoch_orbital_pdos_mse = running_orbital_pdos_mse/n_iter
    epoch_orbital_pdos_mse_cdf = running_orbital_pdos_mse_cdf/n_iter

    error_dict = {'train_loss': epoch_loss,
                  'train_dos_mse': epoch_dos_mse,
                  'train_dos_mse_cdf': epoch_dos_mse_cdf,
                  'train_atomic_dos_mse': epoch_atomic_dos_mse,
                  'train_atomic_dos_mse_cdf': epoch_atomic_dos_mse_cdf,
                  'train_orbital_pdos_mse': epoch_orbital_pdos_mse,
                  'train_orbital_pdos_mse_cdf': epoch_orbital_pdos_mse_cdf}
    
    return error_dict


def validation(model: ProDosNet = None,
               metric: nn.MSELoss = None, 
               epoch: int = None, 
               fold: int = None, 
               save_path: str = None, 
               validation_loader: DataLoader = None, 
               save_output: bool = False,  
               save_dos: bool = False, 
               save_pdos: bool = False, 
               save_id_rmse: bool = False,
               save_id_rmse_interv: int = 50,
               use_cuda: bool = False, 
               use_cdf: bool = False, 
               test: bool = False, 
               train_on_dos: bool = False,
               train_on_atomic_dos: bool = False,
               scaler: Scaler = None) -> dict[str, float]:
    """
        Runs model predictions on validation or test set and returns loss and errors
        ----------------------------------------------------------------------------
        Input:
            - model:                   ProDosNet model
            - metric:                  MSE Loss
            - epoch                    Training epoch
            - fold                     Cross-Valifation fold
            - save_path                Path where outputs will be saved
            - validation_loader        Data loader for validation dataset
            - save_output              If True, saves validation output
            - save_dos                 If True, saves DOS validation predictions
            - save_pdos                If True, saves PDOS validation predictions
            - save_id_rmse             If True, saves validation rmse for each material separately
            - save_id_rmse_interv      Saves rmse for each material every *save_id_rmse_interv* epoch
            - use_cuda                 If True, (GPU is available), loads data to GPU 
            - use_cdf                  If True, uses RMSE Loss of CDFs
            - test                     If True, saves test set outputs
            - scaler                   scale edge features
        Output:
            - epoch_loss
            - epoch_pdos_rmse
            - epoch_cdf_pdos_rmse
        
    """
    model.eval()

    id_error_df_list = []
    if save_output:
        save_dos_list = []
        save_pdos_list = []
        save_dos_list_cdf = []
        save_pdos_list_cdf = []
        
    n_iter = len(validation_loader)

    running_loss = 0.0
    running_dos_mse = 0.0
    running_dos_mse_cdf = 0.0
    running_atomic_dos_mse = 0.0
    running_atomic_dos_mse_cdf = 0.0
    running_orbital_pdos_mse = 0.0
    running_orbital_pdos_mse_cdf = 0.0
    idx_batch = 0
    for data in tqdm(validation_loader, disable = not test):
        e_diff = torch.mean(data.e_diff)

        if save_output:
            target_dos_cpu = data.dos
            target_pdos_cpu = data.pdos
            target_dos_cpu_cdf = data.dos_cdf
            target_pdos_cpu_cdf = data.pdos_cdf
                
        if use_cuda:
            device = torch.device('cuda')
            data = data.to(device)

        if train_on_dos:
            if use_cdf: 
                if scaler is not None:
                    edge_attr = scaler.norm(data.edge_attr)
                else: 
                    edge_attr = data.edge_attr
                
                target_dos = data.dos_cdf
                target_atomic_dos = data.atomic_dos_cdf
                target_orbital_pdos = data.pdos_cdf
         
                output_pdos, output_atomic_dos, output_dos = model(data.x, data.edge_index, edge_attr, data.batch, data.atoms_batch, use_cdf=use_cdf, train_on_pdos=False)

                loss = metric(output_dos/torch.max(target_dos, dim=1)[0].unsqueeze(1), target_dos/torch.max(target_dos, dim=1)[0].unsqueeze(1))
                loss_item = loss.item()
                
                output_dos_diff = torch.diff(output_dos, dim=1)/e_diff
                output_dos_diff[output_dos_diff<0] = 0.0
                dos_mse = mse_loss(output_dos_diff, (torch.diff(target_dos, dim=1)/e_diff)).item()
                dos_mse_cdf = mse_loss(output_dos/torch.max(target_dos, dim=1)[0].unsqueeze(1), target_dos/torch.max(target_dos, dim=1)[0].unsqueeze(1)).item()

                output_atomic_dos_diff = torch.diff(output_atomic_dos, dim=1)/e_diff
                output_atomic_dos_diff[output_atomic_dos_diff<0] = 0.0
                atomic_dos_mse = mse_loss(output_atomic_dos_diff, (torch.diff(target_atomic_dos, dim=1)/e_diff)).item()
                atomic_dos_mse_cdf = mse_loss(output_atomic_dos, target_atomic_dos).item()

                output_orbital_pdos_diff = torch.diff(output_pdos, dim=1)/e_diff
                output_orbital_pdos_diff[output_orbital_pdos_diff<0] = 0.0
                orbital_pdos_mse = mse_loss(output_orbital_pdos_diff, (torch.diff(target_orbital_pdos, dim=1)/e_diff)).item()
                orbital_pdos_mse_cdf = mse_loss(output_pdos, target_orbital_pdos).item()

            else:
                if scaler is not None:
                    edge_attr = scaler.norm(data.edge_attr)
                else: 
                    edge_attr = data.edge_attr

                target_dos = data.dos
                target_atomic_dos = data.atomic_dos
                target_orbital_pdos = data.pdos
         
                output_pdos, output_atomic_dos, output_dos = model(data.x, data.edge_index, edge_attr, data.batch, data.atoms_batch, use_cdf=use_cdf, train_on_pdos=False)

                loss = metric(output_dos/torch.max(target_dos, dim=1)[0].unsqueeze(1), target_dos/torch.max(target_dos, dim=1)[0].unsqueeze(1))
                loss_item = loss.item()
            
                dos_mse = mse_loss(output_dos, target_dos).item()
                
                output_dos_cdf = torch.cumsum(output_dos, dim=1)*e_diff
                target_dos_cdf = torch.cumsum(target_dos, dim=1)*e_diff
                dos_mse_cdf = mse_loss(output_dos_cdf/torch.max(target_dos_cdf, dim=1)[0].unsqueeze(1), target_dos_cdf/torch.max(target_dos_cdf, dim=1)[0].unsqueeze(1)).item()

                atomic_dos_mse = mse_loss(output_atomic_dos, target_atomic_dos).item()
                atomic_dos_mse_cdf = mse_loss(torch.cumsum(output_atomic_dos, dim=1)*e_diff, torch.cumsum(target_atomic_dos, dim=1)*e_diff).item()

                orbital_pdos_mse = mse_loss(output_pdos, target_orbital_pdos).item()
                orbital_pdos_mse_cdf = mse_loss(torch.cumsum(output_pdos, dim=1)*e_diff, torch.cumsum(target_orbital_pdos, dim=1)*e_diff).item()

        elif train_on_atomic_dos:
            if use_cdf: 
                if scaler is not None:
                    edge_attr = scaler.norm(data.edge_attr)
                else: 
                    edge_attr = data.edge_attr
                
                target_dos = data.dos_cdf
                target_atomic_dos = data.atomic_dos_cdf
                target_orbital_pdos = data.pdos_cdf
         
                output_pdos, output_atomic_dos, output_dos = model(data.x, data.edge_index, edge_attr, data.batch, data.atoms_batch, use_cdf=use_cdf, train_on_pdos=False)

                loss = metric(target_atomic_dos, output_atomic_dos)
                loss_item = loss.item()
                
                output_dos_diff = torch.diff(output_dos, dim=1)/e_diff
                output_dos_diff[output_dos_diff<0] = 0.0
                dos_mse = mse_loss(output_dos_diff, (torch.diff(target_dos, dim=1)/e_diff)).item()
                dos_mse_cdf = mse_loss(output_dos, target_dos).item()

                output_atomic_dos_diff = torch.diff(output_atomic_dos, dim=1)/e_diff
                output_atomic_dos_diff[output_atomic_dos_diff<0] = 0.0
                atomic_dos_mse = mse_loss(output_atomic_dos_diff, (torch.diff(target_atomic_dos, dim=1)/e_diff)).item()
                atomic_dos_mse_cdf = mse_loss(output_atomic_dos, target_atomic_dos).item()

                output_orbital_pdos_diff = torch.diff(output_pdos, dim=1)/e_diff
                output_orbital_pdos_diff[output_orbital_pdos_diff<0] = 0.0
                orbital_pdos_mse = mse_loss(output_orbital_pdos_diff, (torch.diff(target_orbital_pdos, dim=1)/e_diff)).item()
                orbital_pdos_mse_cdf = mse_loss(output_pdos, target_orbital_pdos).item()

            else:
                if scaler is not None:
                    edge_attr = scaler.norm(data.edge_attr)
                else: 
                    edge_attr = data.edge_attr

                target_dos = data.dos
                target_atomic_dos = data.atomic_dos
                target_orbital_pdos = data.pdos
         
                output_pdos, output_atomic_dos, output_dos = model(data.x, data.edge_index, edge_attr, data.batch, data.atoms_batch, use_cdf=use_cdf, train_on_pdos=False)

                loss = metric(target_atomic_dos, output_atomic_dos)
                loss_item = loss.item()
            
                dos_mse = mse_loss(output_dos, target_dos).item()
                dos_mse_cdf = mse_loss(torch.cumsum(output_dos, dim=1)*e_diff, torch.cumsum(target_dos, dim=1)*e_diff).item()

                atomic_dos_mse = mse_loss(output_atomic_dos, target_atomic_dos).item()
                atomic_dos_mse_cdf = mse_loss(torch.cumsum(output_atomic_dos, dim=1)*e_diff, torch.cumsum(target_atomic_dos, dim=1)*e_diff).item()

                orbital_pdos_mse = mse_loss(output_pdos, target_orbital_pdos).item()
                orbital_pdos_mse_cdf = mse_loss(torch.cumsum(output_pdos, dim=1)*e_diff, torch.cumsum(target_orbital_pdos, dim=1)*e_diff).item()

        else:
            if use_cdf: 
                if scaler is not None:
                    edge_attr = scaler.norm(data.edge_attr)
                else: 
                    edge_attr = data.edge_attr
                
                target_dos = data.dos_cdf
                target_atomic_dos = data.atomic_dos_cdf
                target_orbital_pdos = data.pdos_cdf
         
                output_pdos, output_atomic_dos, output_dos = model(data.x, data.edge_index, edge_attr, data.batch, data.atoms_batch, use_cdf=use_cdf, train_on_pdos=True)

                loss = metric(target_orbital_pdos, output_pdos)
                loss_item = loss.item()
                
                output_dos_diff = torch.diff(output_dos, dim=1)/e_diff
                output_dos_diff[output_dos_diff<0] = 0.0
                dos_mse = mse_loss(output_dos_diff, (torch.diff(target_dos, dim=1)/e_diff)).item()
                dos_mse_cdf = mse_loss(output_dos, target_dos).item()

                output_atomic_dos_diff = torch.diff(output_atomic_dos, dim=1)/e_diff
                output_atomic_dos_diff[output_atomic_dos_diff<0] = 0.0
                atomic_dos_mse = mse_loss(output_atomic_dos_diff, (torch.diff(target_atomic_dos, dim=1)/e_diff)).item()
                atomic_dos_mse_cdf = mse_loss(output_atomic_dos, target_atomic_dos).item()

                output_orbital_pdos_diff = torch.diff(output_pdos, dim=1)/e_diff
                output_orbital_pdos_diff[output_orbital_pdos_diff<0] = 0.0
                orbital_pdos_mse = mse_loss(output_orbital_pdos_diff, (torch.diff(target_orbital_pdos, dim=1)/e_diff)).item()
                orbital_pdos_mse_cdf = mse_loss(output_pdos, target_orbital_pdos).item()
                   
            else:
                if scaler is not None:
                    edge_attr = scaler.norm(data.edge_attr)
                else: 
                    edge_attr = data.edge_attr

                target_dos = data.dos
                target_atomic_dos = data.atomic_dos
                target_orbital_pdos = data.pdos
         
                output_pdos, output_atomic_dos, output_dos = model(data.x, data.edge_index, edge_attr, data.batch, data.atoms_batch, use_cdf=use_cdf, train_on_pdos=True)

                loss = metric(target_orbital_pdos, output_pdos)
                loss_item = loss.item()
            
                dos_mse = mse_loss(output_dos, target_dos).item()
                dos_mse_cdf = mse_loss(torch.cumsum(output_dos, dim=1)*e_diff, torch.cumsum(target_dos, dim=1)*e_diff).item()

                atomic_dos_mse = mse_loss(output_atomic_dos, target_atomic_dos).item()
                atomic_dos_mse_cdf = mse_loss(torch.cumsum(output_atomic_dos, dim=1)*e_diff, torch.cumsum(target_atomic_dos, dim=1)*e_diff).item()

                orbital_pdos_mse = mse_loss(output_pdos, target_orbital_pdos).item()
                orbital_pdos_mse_cdf = mse_loss(torch.cumsum(output_pdos, dim=1)*e_diff, torch.cumsum(target_orbital_pdos, dim=1)*e_diff).item()

        running_loss += loss_item

        running_dos_mse += dos_mse
        running_dos_mse_cdf += dos_mse_cdf

        running_atomic_dos_mse += atomic_dos_mse
        running_atomic_dos_mse_cdf += atomic_dos_mse_cdf

        running_orbital_pdos_mse += orbital_pdos_mse
        running_orbital_pdos_mse_cdf += orbital_pdos_mse_cdf
        
        if save_id_rmse and epoch%save_id_rmse_interv == 0:
            ids_list = []
            for orbitals, id in zip(data.orbital_types, data.material_id):
                ids_list.append([id]*len(orbitals))

            ids_list = list(itertools.chain.from_iterable(ids_list))
            sites = list(itertools.chain.from_iterable(data.sites))
            elements = list(itertools.chain.from_iterable(data.elements))
            orbital_types = list(itertools.chain.from_iterable(data.orbital_types))

            output_pdos_orb_cdf = output_pdos.reshape(len(ids_list), 256)
            target_pdos_orb_cdf = target_orbital_pdos.reshape(len(ids_list), 256)
            rmse_orb_cdf = ((output_pdos_orb_cdf-target_pdos_orb_cdf)**2).sum(dim=1).sqrt().cpu().detach().numpy()


            output_orbital_pdos_diff = torch.diff(output_pdos, dim=1)/e_diff
            output_orbital_pdos_diff[output_orbital_pdos_diff<0] = 0.0
            target_pdos_diff = torch.diff(target_atomic_dos, dim=1)/e_diff
            
            output_pdos_orb = output_orbital_pdos_diff.reshape(len(ids_list), 255)
            target_pdos_orb = target_pdos_diff.reshape(len(ids_list), 255)
            rmse_orb = ((output_pdos_orb-target_pdos_orb)**2).sum(dim=1).sqrt().cpu().detach().numpy()

            id_error_df = pd.DataFrame({"id": ids_list, "element": elements, "site": sites, "orbital": orbital_types, "rmse": rmse_orb, "rmse_cdf": rmse_orb_cdf})
            id_error_df_list.append(id_error_df)
  
        if save_output:
            if use_cdf:
                dos_to_save = torch.diff(output_dos, dim=1)/e_diff
                dos_to_save_cdf = output_dos
            else:
                dos_to_save = output_dos
                dos_to_save_cdf = torch.cumsum(output_dos, dim=1)*e_diff
            
            # DOS dataframes to save
            out_dos_data = pd.DataFrame(dos_to_save.data.cpu().detach().numpy())
            out_dos_data_cdf = pd.DataFrame(dos_to_save_cdf.data.cpu().detach().numpy())
            target_dos_data = pd.DataFrame(target_dos_cpu.detach().numpy())
            target_dos_data_cdf = pd.DataFrame(target_dos_cpu_cdf.detach().numpy())
            # material id and DOS 
            output_and_id = pd.concat([pd.DataFrame(data.material_id), out_dos_data, target_dos_data], axis = 1, ignore_index=True, sort=False)
            output_and_id_cdf = pd.concat([pd.DataFrame(data.material_id), out_dos_data_cdf, target_dos_data_cdf], axis = 1, ignore_index=True, sort=False)
            save_dos_list.append(output_and_id)
            save_dos_list_cdf.append(output_and_id_cdf)

            if save_pdos:
                if use_cdf:
                    pdos_to_save = output_orbital_pdos_diff
                    pdos_to_save_cdf = output_pdos
                    target_pdos_cpu = (torch.diff(target_orbital_pdos, dim=1)/e_diff).data.cpu()
                    target_pdos_cpu_cdf = target_orbital_pdos
                else:
                    pdos_to_save = output_pdos
                    pdos_to_save_cdf = torch.cumsum(output_pdos, dim=1)*e_diff
                
                ids_list = []
                for orbitals, id in zip(data.orbital_types, data.material_id):
                    ids_list.append([id]*len(orbitals))

                ids_list = list(itertools.chain.from_iterable(ids_list))
                sites = list(itertools.chain.from_iterable(data.sites))
                elements = list(itertools.chain.from_iterable(data.elements))
                orbital_types = list(itertools.chain.from_iterable(data.orbital_types))

                out_pdos_data = pd.DataFrame(pdos_to_save.data.cpu().detach().numpy())
                out_pdos_data_cdf = pd.DataFrame(pdos_to_save_cdf.data.cpu().detach().numpy())
                target_pdos_data = pd.DataFrame(target_pdos_cpu.detach().numpy())
                target_pdos_data_cdf = pd.DataFrame(target_pdos_cpu_cdf.detach().numpy())

                output_and_id = pd.concat([pd.DataFrame(ids_list), pd.DataFrame(elements), pd.DataFrame(sites), pd.DataFrame(orbital_types), out_pdos_data, target_pdos_data], axis = 1, ignore_index=True, sort=False)
                output_and_id_cdf = pd.concat([pd.DataFrame(ids_list), pd.DataFrame(elements), pd.DataFrame(sites), pd.DataFrame(orbital_types), out_pdos_data_cdf, target_pdos_data_cdf], axis = 1, ignore_index=True, sort=False)
                save_pdos_list.append(output_and_id)
                save_pdos_list_cdf.append(output_and_id_cdf)
            
    if save_dos:
        total_output_dos = pd.concat(save_dos_list)
        total_output_dos_cdf = pd.concat(save_dos_list_cdf)
        filename = f'best_model_output_total_dos_fold_{fold}.csv'
        filename_cdf = f'best_model_output_total_dos_cdf_fold_{fold}.csv'
        if test:
            filename = f'test_output_total_dos.csv'
            filename_cdf = f'test_output_total_dos_cdf.csv'
        total_output_dos.to_csv('%s/'%save_path + filename, header=False, index=False)
        total_output_dos_cdf.to_csv('%s/'%save_path + filename_cdf, header=False, index=False)

    if save_pdos:
        total_output_pdos = pd.concat(save_pdos_list)
        total_output_pdos_cdf = pd.concat(save_pdos_list_cdf)
        filename = f'best_model_output_pdos_fold_{fold}.csv'
        filename_cdf = f'best_model_output_pdos_cdf_fold_{fold}.csv'
        if test:
            filename = f'test_output_pdos.csv'
            filename_cdf = f'test_output_pdos_cdf.csv'
        total_output_pdos.to_csv('%s/'%save_path + filename, header=False, index=False)
        total_output_pdos_cdf.to_csv('%s/'%save_path + filename_cdf, header=False, index=False)

    epoch_loss = running_loss/n_iter
    epoch_dos_mse = running_dos_mse/n_iter
    epoch_dos_mse_cdf = running_dos_mse_cdf/n_iter

    epoch_atomic_dos_mse = running_atomic_dos_mse/n_iter
    epoch_atomic_dos_mse_cdf = running_atomic_dos_mse_cdf/n_iter

    epoch_orbital_pdos_mse = running_orbital_pdos_mse/n_iter
    epoch_orbital_pdos_mse_cdf = running_orbital_pdos_mse_cdf/n_iter
    
    error_dict = {'val_loss': epoch_loss,
                  'val_dos_mse': epoch_dos_mse,
                  'val_dos_mse_cdf': epoch_dos_mse_cdf,
                  'val_atomic_dos_mse': epoch_atomic_dos_mse,
                  'val_atomic_dos_mse_cdf': epoch_atomic_dos_mse_cdf,
                  'val_orbital_pdos_mse': epoch_orbital_pdos_mse,
                  'val_orbital_pdos_mse_cdf': epoch_orbital_pdos_mse_cdf}

    if save_id_rmse and epoch%save_id_rmse_interv == 0:
        id_error_total_df = pd.concat(id_error_df_list)
        id_error_total_df.to_csv('%s/'%save_path + f"orbital_rmse_epoch_{epoch}_fold_{fold}.csv", index=False)
    
    return error_dict