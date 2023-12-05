import os
import matplotlib.pyplot as plt
import seaborn as sns
from torch import device, argmax, sum, Tensor, no_grad, logical_and, zeros_like
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

class Evaluator():
    def __init__(self, device, batch_size, model):
        self.device = device
        self.batch_size = batch_size
        self.model = model

    def somma_valori_posizioni(lista):
        sum1 = 0
        sum2 = 0
        sum3 = 0
        for sublist in lista:
            if len(sublist) >= 2:  # Assicurati che ci siano almeno tre elementi nella sottolista
                sum1 += sublist[0] 
                sum2 += sublist[1]
                sum3 += sublist[2]  # Somma gli elementi alle posizioni 1, 2 e 3 (0-based)
        return (sum1, sum2, sum3)
    
    def evaluate_3_heads_classic(self, file_path, dataLoader, class_dictionary_inverted_extended, all_classes):
        correct = 0
        correct_per_class = (0,0,0)
        total1 = 0
        total2 = 0
        total3 = 0
        true_labels = []
        predicted_labels = []
        two_on_three_acc = 0
        
        with no_grad():
            for sample_batched in tqdm(dataLoader):
                inputs = sample_batched['image'].float().to(self.device)
                labels1 = sample_batched['label'][0].long().to(self.device)
                labels2 = sample_batched['label'][1].long().to(self.device)
                labels3 = sample_batched['label'][2].long().to(self.device)

                #outputs = model(inputs, one_hot_input)
                out1, out2, out3 = self.model(inputs)
                # loss = criterion(outputs, labels)
                prob1 = nn.Softmax(dim=1)(out1)
                prob2 = nn.Softmax(dim=1)(out2)
                prob3 = nn.Softmax(dim=1)(out3)

                pred1 = argmax(prob1,dim=1)
                pred2 = argmax(prob2,dim=1)
                pred3 = argmax(prob3,dim=1)

                mask1 = logical_and(logical_and(logical_and(logical_and((labels2 != 3 ), ( labels2 != 4 )), ( labels2 != 5 )) , ( labels3 != 2 )), ( labels3 != 3))  
                mask3 = logical_and(logical_and((labels2 != 3 ),  ( labels2 != 4 )),  ( labels2 != 5))
                
                preds_for_matrix = Tensor([list(pred1),list(pred2),list(pred3)]).clone()
                label_for_matrix = Tensor([list(labels1),list(labels2),list(labels3)]).clone()
                # statistics
                labels1[~mask1] = 100
                labels3[~mask3] = 100
                pred1[~mask1] = labels1[~mask1]
                pred3[~mask3] = labels3[~mask3]
                correctness = (logical_and(logical_and(pred1 == labels1, pred2 == labels2), pred3 == labels3)).long()
                
                total1 += inputs.size(0)
                total2 += inputs.size(0)
                total3 += inputs.size(0)
                correct += sum(correctness)

                classication_per_subclass = []
                labels = []
                predictions = []
                # Confronto degli elementi dei tensori e creazione della lista
                for i in range(len(pred1)):
                    elem1 = 1 if pred1[i] == labels1[i] else 0
                    elem2 = 1 if pred2[i] == labels2[i] else 0
                    elem3 = 1 if pred3[i] == labels3[i] else 0
                    nuova_sublista = [elem1, elem2, elem3]
                    classication_per_subclass.append(nuova_sublista)
                    
                    label = f"{int(label_for_matrix[0][i])}-{int(label_for_matrix[1][i])}-{int(label_for_matrix[2][i])}"
                    predict = f"{int(preds_for_matrix[0][i])}-{int(preds_for_matrix[1][i])}-{int(preds_for_matrix[2][i])}"
                    labels.append(label)
                    predictions.append(predict)
                    if labels3[i] == 100:
                        if nuova_sublista[1] == 1:
                            two_on_three_acc += 1
                    elif labels1[i] == 100:
                        if nuova_sublista[1] == 1 and nuova_sublista[2] == 1:
                            two_on_three_acc += 1
                    else:
                        if __builtins__.sum(nuova_sublista) > 1:
                            two_on_three_acc += 1
                correct_per_class = list(x + y for x, y in zip(correct_per_class , self.somma_valori_posizioni(classication_per_subclass)))
                correct_per_class[0] -= len(pred1[~mask1])
                correct_per_class[2] -= len(pred3[~mask3])
                total1 -= len(pred1[~mask1])
                total3 -= len(pred3[~mask3])
                true_labels.extend(np.array(labels))
                predicted_labels.extend(np.array(predictions))
        true_labels_trad = [class_dictionary_inverted_extended[item] for item in true_labels]
        predicted_labels_trad = []
        for item in predicted_labels:
            try:
                predicted_labels_trad.append(class_dictionary_inverted_extended[item])
            except:
                predicted_labels_trad.append("27")
        # Extract the file name without extension
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        # Calcola la confusion matrix
        confusion = confusion_matrix(true_labels_trad, predicted_labels_trad)


        # Crea la figura e l'asse
        fig, ax = plt.subplots(figsize=(30, 20))

        # Crea la heatmap con annotazioni
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=all_classes, yticklabels=all_classes, ax=ax)

        # Imposta i titoli e le etichette degli assi
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title("Confusion Matrix")

        # Salva la figura
        plt.savefig('/user/mmarseglia/first_split_as_paper/all_matrices/'+file_name+'.png')

        # Calcola l'accuratezza
        accuracy = 100 * correct / total2
        acc_per_class =  tuple((x/y)*100 for x, y in zip(correct_per_class , (total1, total2, total3)))
        tt_acc = (two_on_three_acc/total2)*100

        # Return the results along with the file name
        return {
            'File Name': file_name,
            'General Accuracy': accuracy,
            'Two on Three Accuracy': tt_acc,
            'Accuracy per Class': acc_per_class,
        }
        
    def evaluate_single_head_classic(self, file_path, dataLoader, class_dictionary_inverted, class_dictionary, all_classes):
        correct = 0
        correct_per_class = (0,0,0)
        total1 = 0
        total2 = 0
        total3 = 0
        total = 0
        true_labels = []
        predicted_labels = []
        two_on_three_acc = 0

        with no_grad():
            for sample_batched in tqdm(dataLoader):
                inputs = sample_batched['image'].float().to(device)
                labels = sample_batched['label'].long().to(device)

                outputs = self.model(inputs)
                prob = nn.Softmax(dim=1)(outputs)
                predictions = argmax(prob, dim=1)

                total += labels.size(0)
                correct += (predictions == labels).sum().item()

                # Lista delle stringhe corrispondenti agli indici del tensore
                pred_strings = Tensor([class_dictionary[str(index.item())] for index in predictions])
                lab_strings = Tensor([class_dictionary[str(index.item())] for index in labels])
                pred1 = pred_strings[:, 0]
                pred2 = pred_strings[:, 1]
                pred3 = pred_strings[:, 2]
                
                labels1 = lab_strings[:, 0]
                labels2 = lab_strings[:, 1]
                labels3 = lab_strings[:, 2]
                            
                mask1 = logical_and(logical_and(logical_and(logical_and((labels2 != 3 ), ( labels2 != 4 )), ( labels2 != 5 )) , ( labels3 != 2 )), ( labels3 != 3))  
                mask3 = logical_and(logical_and((labels2 != 3 ),  ( labels2 != 4 )),  ( labels2 != 5))
                #save the values before modify them
                preds_for_matrix = pred_strings.clone()
                # statistics
                labels1[~mask1] = 100
                labels3[~mask3] = 100
                pred1[~mask1] = labels1[~mask1]
                pred3[~mask3] = labels3[~mask3]

                total1 += inputs.size(0)
                total2 += inputs.size(0)
                total3 += inputs.size(0)

                classication_per_subclass = []
                labels = []
                all_predictions = []
                # Confronto degli elementi dei tensori e creazione della lista
                for i in range(len(pred1)):
                    elem1 = 1 if pred1[i] == labels1[i] else 0
                    elem2 = 1 if pred2[i] == labels2[i] else 0
                    elem3 = 1 if pred3[i] == labels3[i] else 0
                    nuova_sublista = [elem1, elem2, elem3]
                    classication_per_subclass.append(nuova_sublista)
                    
                    label = class_dictionary_inverted[f"{int(labels1[i])}-{int(labels2[i])}-{int(labels3[i])}"]
                    predict = class_dictionary_inverted[f"{int(preds_for_matrix[i][0])}-{int(preds_for_matrix[i][1])}-{int(preds_for_matrix[i][2])}"]
                    labels.append(label)
                    all_predictions.append(predict)
                    if labels3[i] == 100:
                        if nuova_sublista[1] == 1:
                            two_on_three_acc += 1
                    elif labels1[i] == 100:
                        if nuova_sublista[1] == 1 and nuova_sublista[2] == 1:
                            two_on_three_acc += 1
                    else:
                        if __builtins__.sum(nuova_sublista) > 1:
                            two_on_three_acc += 1
                correct_per_class = list(x + y for x, y in zip(correct_per_class , self.somma_valori_posizioni(classication_per_subclass)))
                correct_per_class[0] -= len(pred1[~mask1])
                correct_per_class[2] -= len(pred3[~mask3])
                total1 -= len(pred1[~mask1])
                total3 -= len(pred3[~mask3])
                true_labels.extend(np.array(labels))
                predicted_labels.extend(np.array(all_predictions))

        # Extract the file name without extension
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        # Calcola la confusion matrix
        confusion = confusion_matrix(true_labels, predicted_labels)


        # Crea la figura e l'asse
        fig, ax = plt.subplots(figsize=(30, 20))

        # Crea la heatmap con annotazioni
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=all_classes, yticklabels=all_classes, ax=ax)

        # Imposta i titoli e le etichette degli assi
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title("Confusion Matrix")

        # Salva la figura
        plt.savefig('/user/mmarseglia/first_split_as_paper/all_matrices/'+file_name+'.png')

        # Calcola l'accuratezza
        accuracy = 100 * correct / total
        acc_per_class =  tuple((x/y)*100 for x, y in zip(correct_per_class , (total1, total2, total3)))
        tt_acc = (two_on_three_acc/total)*100



        # Return the results along with the file name
        return {
            'File Name': file_name,
            'General Accuracy': accuracy,
            'Two on Three Accuracy': tt_acc,
            'Accuracy per Class': acc_per_class
        }

    def evaluate_3_heads_history(self, file_path, dataLoader, class_dictionary_inverted_extended, all_classes):
        correct = 0
        correct_per_class = (0,0,0)
        total1 = 0
        total2 = 0
        total3 = 0
        true_labels = []
        predicted_labels = []
        two_on_three_acc = 0
        
        with no_grad():
            for sample_batched in tqdm(dataLoader):
                inputs = sample_batched['image'].float().to(device)
                previous_labels = sample_batched['previous_classes']  # Una lista di liste di classificazioni precedenti
                labels1 = sample_batched["current_class"][0].long().to(device)
                labels2 = sample_batched["current_class"][1].long().to(device)
                labels3 = sample_batched["current_class"][2].long().to(device)
                
                out1, current_outputs, out3 = self.model(inputs)
                
                prob1 = nn.Softmax(dim=1)(out1)
                current_probabilities = nn.Softmax(dim=1)(current_outputs)
                prob3 = nn.Softmax(dim=1)(out3)
                
                # Calcola le correzioni in base alle predizioni delle immagini precedenti
                corrections = zeros_like(current_outputs)
                max_distance = len(previous_labels)
                weights = [0.8**(i+1) for i in range(max_distance)]

                for i, previous_class in enumerate(previous_labels):
                    indici_validi = [i for i, elemento in enumerate(previous_class) if elemento != "not_usable"]
                    previous_inputs = sample_batched['previous_images'][i][indici_validi].float().to(device)
                    out1, previous_outputs, out3 = self.model(previous_inputs)
                    previous_probabilities = nn.Softmax(dim=1)(previous_outputs)
                    corrections[indici_validi] += weights[i] * (previous_probabilities - current_probabilities[indici_validi])

                # Applica le correzioni all'output corrente
                corrected_outputs = current_probabilities + corrections            

                pred1 = argmax(prob1,dim=1)
                pred2 = argmax(corrected_outputs,dim=1)
                pred3 = argmax(prob3,dim=1)

                mask1 = logical_and(logical_and(logical_and(logical_and((labels2 != 3 ), ( labels2 != 4 )), ( labels2 != 5 )) , ( labels3 != 2 )), ( labels3 != 3))  
                mask3 = logical_and(logical_and((labels2 != 3 ),  ( labels2 != 4 )),  ( labels2 != 5))
        
                preds_for_matrix = Tensor([list(pred1),list(pred2),list(pred3)]).clone()
                label_for_matrix = Tensor([list(labels1),list(labels2),list(labels3)]).clone()
                # statistics
                labels1[~mask1] = 100
                labels3[~mask3] = 100
                pred1[~mask1] = labels1[~mask1]
                pred3[~mask3] = labels3[~mask3]
                correctness = (logical_and(logical_and(pred1 == labels1, pred2 == labels2), pred3 == labels3)).long()
                
                total1 += inputs.size(0)
                total2 += inputs.size(0)
                total3 += inputs.size(0)
                correct += sum(correctness)

                classication_per_subclass = []
                labels = []
                predictions = []
                # Confronto degli elementi dei tensori e creazione della lista
                for i in range(len(pred1)):
                    elem1 = 1 if pred1[i] == labels1[i] else 0
                    elem2 = 1 if pred2[i] == labels2[i] else 0
                    elem3 = 1 if pred3[i] == labels3[i] else 0
                    nuova_sublista = [elem1, elem2, elem3]
                    classication_per_subclass.append(nuova_sublista)
                    
                    label = f"{int(label_for_matrix[0][i])}-{int(label_for_matrix[1][i])}-{int(label_for_matrix[2][i])}"
                    predict = f"{int(preds_for_matrix[0][i])}-{int(preds_for_matrix[1][i])}-{int(preds_for_matrix[2][i])}"
                    labels.append(label)
                    predictions.append(predict)
                    if labels3[i] == 100:
                        if nuova_sublista[1] == 1:
                            two_on_three_acc += 1
                    elif labels1[i] == 100:
                        if nuova_sublista[1] == 1 and nuova_sublista[2] == 1:
                            two_on_three_acc += 1
                    else:
                        if __builtins__.sum(nuova_sublista) > 1:
                            two_on_three_acc += 1
                correct_per_class = list(x + y for x, y in zip(correct_per_class , self.somma_valori_posizioni(classication_per_subclass)))
                correct_per_class[0] -= len(pred1[~mask1])
                correct_per_class[2] -= len(pred3[~mask3])
                total1 -= len(pred1[~mask1])
                total3 -= len(pred3[~mask3])
                true_labels.extend(np.array(labels))
                predicted_labels.extend(np.array(predictions))
        true_labels_trad = [class_dictionary_inverted_extended[item] for item in true_labels]
        predicted_labels_trad = []
        for item in predicted_labels:
            try:
                predicted_labels_trad.append(class_dictionary_inverted_extended[item])
            except:
                predicted_labels_trad.append("27")
        # Extract the file name without extension
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        # Calcola la confusion matrix
        confusion = confusion_matrix(true_labels_trad, predicted_labels_trad)


        # Crea la figura e l'asse
        fig, ax = plt.subplots(figsize=(30, 20))

        # Crea la heatmap con annotazioni
        sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=all_classes, yticklabels=all_classes, ax=ax)

        # Imposta i titoli e le etichette degli assi
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title("Confusion Matrix")

        # Salva la figura
        plt.savefig('/user/mmarseglia/first_split_as_paper/all_matrices/'+file_name+'.png')

        # Calcola l'accuratezza
        accuracy = 100 * correct / total2
        acc_per_class =  tuple((x/y)*100 for x, y in zip(correct_per_class , (total1, total2, total3)))
        tt_acc = (two_on_three_acc/total2)*100

        # Return the results along with the file name
        return {
            'File Name': file_name,
            'General Accuracy': accuracy,
            'Two on Three Accuracy': tt_acc,
            'Accuracy per Class': acc_per_class,
        }
