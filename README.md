# MMDTA

MMDTA: Multimodal deep model for drug-target binding affinity with hybrid fusion strategy

## Requirements

[numpy](https://numpy.org/)==1.23.5

[pandas](https://pandas.pydata.org/)==1.5.2

[biopython](https://biopython.org/)==1.79

[scipy](https://scipy.org/)==1.9.3

[torch](https://pytorch.org/)==2.0.1

[torch_geometric]([PyG Documentation — pytorch_geometric documentation (pytorch-geometric.readthedocs.io)](https://pytorch-geometric.readthedocs.io/en/latest/index.html))==2.3.1

## Example usage

### 1. Use our pre-trained model
In this section，we provide the core set data of pdbbindv2016, you can directly execute the following command to run our pre-trained model and get the results on the core set.
```bash
# Run the following command.
python test_pretrain.py
```

### 2. Run on your datasets

In this section, you must provide .sdf file of the drug as well as .pdb file of the target. Note: due to the file "distance_matrix" is too large for us to upload to the github warehouse. You can use the provided code to generate "distance_ Matrix" file, or go to the following link to download: https://pan.baidu.com/s/1UJz1mSW5dZmPNAxWPd8yxA    Extracted Code：u6u5

 ```bash
# You can get the drug map representation and the target distance matrix by running the following command.
python create_drug_graph.py
python create_target_distance_matrix.py

# When all the data is ready, you can train your own model by running the following command.
python training.py

 ```
