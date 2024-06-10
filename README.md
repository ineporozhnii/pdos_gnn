# Graph Neural Network for Projected Density of States Prediction 

In this work, we developed a Graph Neural Network (GNN) ProDosNet which is trained on orbital Projected Density of States (PDOS) data and capable of predicting the electronic structure of materials at extremely low computational cost. With this model, we were able to generate PDOS fingerprints for all compounds present in the Materials Projects Database and cluster them by the similarity of their orbital PDOS, and therefore electronic properties. We demonstrate that using PDOS fingerprints allows finding materials that have similar electronic properties but drastically different structures.

## The model is available via web application: [ProDosMate](https://huggingface.co/spaces/inep/prodosmate "ProDosMate")
### Predict the Projected Density of States
<p align="center">
  <img src="https://github.com/ineporozhnii/pdos_gnn/blob/main/assets/ProDosMate_demo_predict_pdos.gif" alt="animated" />
</p>

### Find materials with similar PDOS
### Explore material space structured by PDOS similarity 

## Setup locally
1. Clone the repository
    - `git clone git@github.com:ineporozhnii/pdos_gnn.git`
2. Create a virtual environment:
    - `cd pdos_gnn/`
    - `python -m venv pdos_gnn_env`
    - `source pdos_gnn_env/bin/activate`
3. To install project run: 
    - `pip install -e .`
  
## Run locally
1. Preprocess data for training from raw Materials Project PDOS
  - `python main.py --task preprocess  --preprocess_ids path/to/ids.csv --cif_dir path/to/cif_files --dos_dir path/to/raw_dos_files`
2. Run training
  - `python main.py --task cross_val  --train_ids path/to/train_ids.csv --data_file path/to/processed_data.tar`
3. Predict using a pre-trained model
  - `python main.py --task test  --test_ids path/to/test_ids.csv --data_file path/to/processed_data.tar --model path/to/pretrained_pdos_model.pth.tar`
