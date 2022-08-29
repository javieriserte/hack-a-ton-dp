## Disorder and linker datasets from Disprot

The purpose of this project is to create **pytorch** datasets for the **Disorder** and **Linker** datasets from [Disprot](https://disprot.org/).

## Requirements

* **python 3.8+**
* **pytorch**
* **torchvision**
* **pandas**
* **matplotlib**
* **numpy**
* **biopython**
* **tqdm**

## Structure of the project
    /data
        /clusters   -> contains the clusterization results created with CD-HIT
        /dataset    -> contains the json files with the disorder and linker datasets (id, sequence and target)
        /fasta      -> contains the fasta files of the sequences in the two datasets (disorder and linker)
        /references -> contains the references (targets) in fasta-like format
        /splits     -> contains the splits of the datasets in train and test (uniprot and disprot id)
        /features
            /pssm   -> results of PSI-BLAST on uniref50
    /dataset        -> contains python code to create the datasets for Pytorch
            disprot_dataset.py -> Contains the code for the creation of the two datasets (disorder and linker)
            encoding.py        -> Contains the code for the encoding of the string sequences in the datasets
            utils.py           -> Contains utils code to read pssm and encode the targets
    /scripts             -> contains bash scripts to run jobs to extract features (hhblits, spider2, psi-blast)
    crate_references.py  -> contains the code to create the references in fasta-like format
    create_train_test.py -> contains the code to run CD-HIT and to create the splits in train and test (uniprot and disprot id) 
    parse_features.py    -> contains the code to parse the features (hhblits, spider2, psi-blast) - not used for now
    main.py              -> contains the code to create the network model, train the model and test the model

## What has been already done

Starting from Disprot two datasets (disorder and linker) were created in this way:

- First the uniprot sequences of the related annotations were extracted, filtering the sequences longer than 4000 residues
- Subsequently a 30% sequence identity clustering was done via CD-HIT
- Starting then from the clusters thus obtained, these were divided by counting 70% of the sequences for the training set and 30% for the test set.
- The datasets containing the sequence id, the sequence and the reference (the true positive and true negative in 0/1 format) were then created. The datasets were exported in json format.
- Different features have been calculated for all the sequences of the two datasets (PSSM, HHM, SPIDER2)

Regarding the DataLoader (or rather Dataset) for PyTorch the following things have been done:

- The basic data (sequence and target) are loaded from the previously created json
- The sequence is transformed by encoding into an integer tensor (pytorch does not support strings), this can already be used as a feature if desired
- Also, a parameter can be passed to load the PSSM feature to have a total of 21 features for each residue (one to identify the residual and 20 of PSSM)
