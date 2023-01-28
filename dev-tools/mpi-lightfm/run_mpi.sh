#!/usr/bin/env bash

for LOSS in bpr warp
do
  for NUM_WORKERS in 5 10 20
      do

      mpirun \
      --mca btl tcp,self \
      --mca btl_base_verbose 20 \
      --mca oob_base_verbose 10 \
      -n $NUM_WORKERS python3 run_experiment.py \
          --loss $LOSS \
          --num_mpi_workers $NUM_WORKERS \
          --path_to_train_interactions_csr train-10m.npz \
          --path_to_test_interactions_csr test-10m.npz \
          --path_to_prediction_user_item_pairs prediction_pairs-10m.csv \
          --dataset_name 10m \
          --num_epochs 30
      done

  mpirun \
      --mca btl tcp,self \
      --mca btl_base_verbose 20 \
      --mca oob_base_verbose 10 \
      -n 40 \
      --oversubscribe python3 run_experiment.py \
          --loss $LOSS \
          --num_mpi_workers 40 \
          --path_to_train_interactions_csr train-10m.npz \
          --path_to_test_interactions_csr test-10m.npz \
          --path_to_prediction_user_item_pairs prediction_pairs-10m.csv \
          --dataset_name 10m \
          --num_epochs 30
done