#!/bin/bash


# Create a log directory with timestamp
log_dir="logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p $log_dir

# For each test: test on all algorithms, on all models

# Default testing parameters: -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR

# Number of clients: test numbers [20, 50, 100]
# Client engagement rate: test numbers [0.6, 0.8, 1.0]
# Learning rate: [0.001, 0.01, 0.1]
# Batch size: [32. 64, 128]
# Local epochs: [1, 5, 10]

# Algorithms: FedAvg, FedDyn, MOON, KT pFL
# Implementations: Multithreading & Model compression

# Function to run experiment and log results
move_files() {
    dirname="$1"
    cd ../results_archive
    mkdir -p "${dirname}"
    cd ../results
    mv * "../results_archive/${dirname}"
    cd ../system
}

move_logs() {
    dirname="$1"
    cd $log_dir
    mv * "../../results_archive/${dirname}"
    cd ..
}

rename_files() {
    old_name="$1"
    new_name="$2"
    cd ../results
    mv "$old_name" "$new_name"
    cd ../system
}

GR=50

## ====
echo "PART 1: PARAMETER TESTING"
## ====

# ====================
# Test 1: Varying number of clients (nc)
    # Test on 20 clients
# #Create the datasets
cd ../dataset
python generate_MNIST.py noniid  - pat 20
# python generate_Cifar10.py noniid - pat 20
cd ../system


# #     # Test on 20 clients
python main.py -algo FedAvg -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/fedavg_nc20.log"
python main.py -algo FedDyn -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR --alpha 0.1 -dev cuda -data MNIST &> "$log_dir/feddyn_nc20.log"
python main.py -algo MOON -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/moon_nc20.log"
python main.py -algo PFL-DA -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/pflda_nc20.log"
move_files "client number 20"
move_logs "client number 20"
# #     # Test on 50 clients
# # #Create the datasets
# cd ../dataset
python generate_MNIST.py noniid - pat 50
# # python generate_Cifar10.py noniid - pat 50
# cd ../system
# # # Test
python main.py -algo FedAvg -nc 50 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/fedavg_nc50.log"
python main.py -algo FedDyn -nc 50 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda --alpha 0.1 -data MNIST &> "$log_dir/feddyn_nc50.log"
python main.py -algo MOON -nc 50 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/moon_nc50.log"
python main.py -algo PFL-DA -nc 50 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/pflda_nc50.log"
move_files "client number 50"
move_logs "client number 50"
# #     # Test on 100 clients
# # #Create the datasets
# cd ../dataset
python generate_MNIST.py noniid - pat 100
# # python generate_Cifar10.py noniid - pat 100
# cd ../system
# # # Test
python main.py -algo FedAvg -nc 100 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/fedavg_nc100.log"
python main.py -algo FedDyn -nc 100 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda --alpha 0.1 -data MNIST &> "$log_dir/feddyn_nc100.log"
python main.py -algo MOON -nc 100 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/moon_nc100.log"
python main.py -algo PFL-DA -nc 100 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/pflda_nc100.log"
move_files "client number 100"
move_logs "client number 100"

# # # ====================
# # # Test 2: Varying join ratios (jr)
# # remake dataset
# cd ../dataset
python generate_MNIST.py noniid - pat 20
# python generate_Cifar10.py noniid - pat 20
# cd ../system
# #     # Test on 0.6
python main.py -algo FedAvg -nc 20 -jr 0.6 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/fedavg_jr0.6.log"
python main.py -algo FedDyn -nc 20 -jr 0.6 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST --alpha 0.1 &> "$log_dir/feddyn_jr0.6.log"
python main.py -algo MOON -nc 20 -jr 0.6 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/moon_jr0.6.log"
python main.py -algo PFL-DA -nc 20 -jr 0.6 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/pflda_jr0.6.log"
move_files "join ratio 0.6"
move_logs "join ratio 0.6"
# #     # Test on 0.8
python main.py -algo FedAvg -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/fedavg_jr0.8.log"
python main.py -algo FedDyn -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST --alpha 0.1 &> "$log_dir/feddyn_jr0.8.log"
python main.py -algo MOON -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/moon_jr0.8.log"
python main.py -algo PFL-DA -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/pflda_jr0.8.log"
move_files "join ratio 0.8"
move_logs "join ratio 0.8"
# #     # Test on 1.0
python main.py -algo FedAvg -nc 20 -jr 1.0 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/fedavg_jr1.0.log"
python main.py -algo FedDyn -nc 20 -jr 1.0 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST --alpha 0.1 &> "$log_dir/feddyn_jr1.0.log"
python main.py -algo MOON -nc 20 -jr 1.0 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/moon_jr1.0.log"
python main.py -algo PFL-DA -nc 20 -jr 1.0 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/pflda_jr1.0.log"
move_files "join ratio 1.0"
move_logs "join ratio 1.0"

# # # ====================
# # # Test 3: Varying learning rates (lr)
# #     # Test on 0.001
python main.py -algo FedAvg -nc 20 -jr 0.8 -lr 0.001 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/fedavg_lr0.001.log"
python main.py -algo FedDyn -nc 20 -jr 0.8 -lr 0.001 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/feddyn_lr0.001.log"
python main.py -algo MOON -nc 20 -jr 0.8 -lr 0.001 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/moon_lr0.001.log"
python main.py -algo PFL-DA -nc 20 -jr 0.8 -lr 0.001 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/pflda_lr0.001.log"
move_files "learning rate 0.001"
move_logs "learning rate 0.001"
# #     # Test on 0.01
python main.py -algo FedAvg -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/fedavg_lr0.01.log"
python main.py -algo FedDyn -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST --alpha 0.1 &> "$log_dir/feddyn_lr0.01.log"
python main.py -algo MOON -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/moon_lr0.01.log"
python main.py -algo PFL-DA -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/pflda_lr0.01.log"
move_files "learning rate 0.01"
move_logs "learning rate 0.01"
# #     # Test on 0.1
python main.py -algo FedAvg -nc 20 -jr 0.8 -lr 0.1 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/fedavg_lr0.1.log"
python main.py -algo FedDyn -nc 20 -jr 0.8 -lr 0.1 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST --alpha 0.1 &> "$log_dir/feddyn_lr0.1.log"
python main.py -algo MOON -nc 20 -jr 0.8 -lr 0.1 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/moon_lr0.1.log"
python main.py -algo PFL-DA -nc 20 -jr 0.8 -lr 0.1 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/pflda_lr0.1.log"
move_files "learning rate 0.1"
move_logs "learning rate 0.1"

# # # ====================
# # # Test 4: Varying local batch size (lbs)
# #     # Test on 32
python main.py -algo FedAvg -nc 20 -jr 0.8 -lr 0.01 -lbs 32 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/fedavg_lbs32.log"
python main.py -algo FedDyn -nc 20 -jr 0.8 -lr 0.01 -lbs 32 -ls 5 -gr $GR -dev cuda -data MNIST --alpha 0.1 &> "$log_dir/feddyn_lbs32.log"
python main.py -algo MOON -nc 20 -jr 0.8 -lr 0.01 -lbs 32 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/moon_lbs32.log"
python main.py -algo PFL-DA -nc 20 -jr 0.8 -lr 0.01 -lbs 32 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/pflda_lbs32.log"
move_files "local batch size 32"
move_logs "local batch size 32"
# #     # Test on 64
python main.py -algo FedAvg -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/fedavg_lbs64.log"
python main.py -algo FedDyn -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST --alpha 0.1 &> "$log_dir/feddyn_lbs64.log"
python main.py -algo MOON -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/moon_lbs64.log"
python main.py -algo PFL-DA -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/pflda_lbs64.log"
move_files "local batch size 64"
move_logs "local batch size 64"
#     # Test on 128
python main.py -algo FedAvg -nc 20 -jr 0.8 -lr 0.01 -lbs 128 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/fedavg_lbs128.log"
python main.py -algo FedDyn -nc 20 -jr 0.8 -lr 0.01 -lbs 128 -ls 5 -gr $GR -dev cuda -data MNIST --alpha 0.1 &> "$log_dir/feddyn_lbs128.log"
python main.py -algo MOON -nc 20 -jr 0.8 -lr 0.01 -lbs 128 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/moon_lbs128.log"
python main.py -algo PFL-DA -nc 20 -jr 0.8 -lr 0.01 -lbs 128 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/pflda_lbs128.log"
move_files "local batch size 128"
move_logs "local batch size 128"

# # # ====================
# # # Test 5: Varying local steps (ls)
# #     # Test on 1
python main.py -algo FedAvg -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 1 -gr $GR -dev cuda -data MNIST &> "$log_dir/fedavg_ls1.log"
python main.py -algo FedDyn -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 1 -gr $GR -dev cuda -data MNIST &> "$log_dir/feddyn_ls1.log"
python main.py -algo MOON -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 1 -gr $GR -dev cuda -data MNIST &> "$log_dir/moon_ls1.log"
python main.py -algo PFL-DA -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 1 -gr $GR -dev cuda -data MNIST &> "$log_dir/pflda_ls1.log"
move_files "local steps 1"
move_logs "local steps 1"
#     # Test on 5
python main.py -algo FedAvg -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/fedavg_ls5.log"
python main.py -algo FedDyn -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST --alpha 0.1 &> "$log_dir/feddyn_ls5.log"
python main.py -algo MOON -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/moon_ls5.log"
python main.py -algo PFL-DA -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data MNIST &> "$log_dir/pflda_ls5.log"
move_files "local steps 5"
move_logs "local steps 5"
# #     # Test on 10
python main.py -algo FedAvg -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 10 -gr $GR -dev cuda -data MNIST &> "$log_dir/fedavg_ls10.log"
python main.py -algo FedDyn -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 10 -gr $GR -dev cuda -data MNIST --alpha 0.1 &> "$log_dir/feddyn_ls10.log"
python main.py -algo MOON -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 10 -gr $GR -dev cuda -data MNIST &> "$log_dir/moon_ls10.log"
python main.py -algo PFL-DA -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 10 -gr $GR -dev cuda -data MNIST &> "$log_dir/pflda_ls10.log"
move_files "local steps 10"
move_logs "local steps 10"

# # ## ====
echo "PART 2: ALGO/MODEL COMPARISON"
# # ## ====
echo "Trying out more algorithms"
# python main.py -algo FedAvg -nc 20 -r 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data Cifar10 &> "$log_dir/CIFARfedavg.log"
python main.py -algo FedDyn -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 --alpha 0.1 -gr $GR -dev cuda -data Cifar10 &> "$log_dir/CIFARfeddyn.log"
# python main.py -algo MOON -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data Cifar10 &> "$log_dir/CIFARmoon.log"
python main.py -algo PFL-DA -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 5 -gr $GR -dev cuda -data Cifar10 &> "$log_dir/CIFARpflda.log"
python main.py -algo Ditto -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -gr $GR -dev cuda -data Cifar10 &> "$log_dir/CIFARditto.log"
move_files "Cifar Algo Comparison"
move_logs "Cifar Algo Comparison"

echo "Trying out more models"
python main.py -algo FedAvg -m CNN -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -gr $GR -dev cuda -data Cifar10 &> "$log_dir/CNN.log" # WORKS OK
rename_files "Cifar10_FedAvg_test_0.h5" "Cifar10_CNN.h5"
python main.py -algo FedAvg -m LeNetCifar -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -gr $GR -dev cuda -data Cifar10 &> "$log_dir/LeNet.log"
rename_files "Cifar10_FedAvg_test_0.h5" "Cifar10_LeNet.h5"
python main.py -algo FedAvg -m VGG11 -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -gr $GR -dev cuda -data Cifar10 &> "$log_dir/VGG11.log" #WORK ON CIFAR ONLY
rename_files "Cifar10_FedAvg_test_0.h5" "Cifar10_VGG11.h5"
python main.py -algo FedAvg -m ResNet18 -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -gr $GR -dev cuda -data Cifar10 &> "$log_dir/ResNet18.log" #WORK ON CIFAR ONLY
rename_files "Cifar10_FedAvg_test_0.h5" "Cifar10_ResNet18.h5"
move_files "Cifar Model Comparison"
move_logs  "Cifar Model Comparison"


# # ## ====
echo "PART 3: IMRPOVEMENT COMPARISON"
# # ## ====
# # Test on Multithreading + direct comparison
python main.py -algo FedDyn -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 1 -gr $GR -dev cuda -data MNIST &> "$log_dir/feddyn.log"
python main.py -algo FedDynThread -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 1 -gr $GR -dev cuda -data MNIST &> "$log_dir/feddynthread.log"
python main.py -algo MOON -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 1 -gr $GR -dev cuda -data MNIST &> "$log_dir/moon.log"
python main.py -algo MOONThread -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 1 -gr $GR -dev cuda -data MNIST &> "$log_dir/moonthreaded.log"
move_files "threading comparison"
move_logs "threading comparison"

# Test on model compression - testing different compression levels
# echo 'testing compression tests'
python main.py -algo FedAvg -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 1 -gr $GR -dev cuda -data MNIST &> "$log_dir/fedavg_noncompressed.log"
rename_files "MNIST_FedAvg_test_0.h5" "MNIST_FedAvg_test_uncompressed.h5"
python main.py -algo FedAvgPruned -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 1 -gr $GR -dev cuda -data MNIST --pruning_ratio 0.25 --pruning_method magnitude --pruning_frequency 5 &> "$log_dir/fedavg_pruned25.log"
rename_files "MNIST_FedAvgPruned_test_0.h5" "MNIST_FedAvg_test_25.h5"
python main.py -algo FedAvgPruned -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 1 -gr $GR -dev cuda -data MNIST --pruning_ratio 0.5 --pruning_method magnitude --pruning_frequency 5 &> "$log_dir/fedavg_pruned50.log"
rename_files "MNIST_FedAvgPruned_test_0.h5" "MNIST_FedAvg_test_50.h5"
python main.py -algo FedAvgPruned -nc 20 -jr 0.8 -lr 0.01 -lbs 64 -ls 1 -gr $GR -dev cuda -data MNIST --pruning_ratio 0.75 --pruning_method magnitude --pruning_frequency 5 &> "$log_dir/fedavg_pruned75.log"
rename_files "MNIST_FedAvgPruned_test_0.h5" "MNIST_FedAvg_test_75.h5"
move_files "compression comparison"
move_logs "compression comparison"

rmdir $log_dir

echo 'done'
