import json
import matplotlib.pyplot as plt

json_path = "C:/Users/mmsca/Desktop/tesi/repository_tesi/Progetto_Tesi/training_results/reorganized_dataset/paper_split/multihead/complex_multi_head_model_cami_way/more_dropout/MH_SI_dummy_variables_SI_grad_norm_cami_way_1_into_2_into3_more_dropout_paper_split.json"

with open(json_path, 'r') as file:
    data = json.load(file)
    train_data = data['trainData']
    val_data = data['valData']
    num_epoch = data['num epoch']

    curve_acc_train = []
    curve_acc_val = []
    curve_loss_train = []
    curve_loss_val = []
    epochs = []


    for i in range(num_epoch):
        epochs.append(i)

        curve_loss_train.append(train_data[i]["TrainLoss"])
        curve_acc_train.append(train_data[i]["Valacc"])

        curve_loss_val.append(val_data[i]["ValLoss"])
        curve_acc_val.append(val_data[i]["Valacc"])
    
    # LOSS PLOT
    plt.figure(figsize=(8, 3))
    # Plot the first curve
    plt.plot(epochs, curve_loss_train, marker='o', linestyle='-', color='b', label='train')
    # Plot the second curve
    plt.plot(epochs, curve_loss_val, marker='x', linestyle='--', color='r', label='valid')
    # Add labels and title
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Loss among epochs')
    # Add legend
    plt.legend()
    plt.savefig(json_path[:-5] + "_Loss.png")
    

    # ACCURACY PLOT
    plt.figure(figsize=(8, 3))
    # Plot the first curve
    plt.plot(epochs, curve_acc_train, marker='o', linestyle='-', color='b', label='train')
    # Plot the second curve
    plt.plot(epochs, curve_acc_val, marker='x', linestyle='--', color='r', label='valid')

    # Add labels and title
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('Accuracy among epochs')
    # Add legend
    plt.legend()
    plt.savefig(json_path[:-5] + "_Accuracy.png")

    