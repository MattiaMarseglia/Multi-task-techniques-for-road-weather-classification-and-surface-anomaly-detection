# trainer.py
import json
import torch.nn as nn
from tqdm import tqdm
import time
import requests
import copy
import torch
import numpy as np

class Trainer():
    def __init__(self, model ,weigths_file_name, dataloaders, dataset_sizes, batch_size):
        self.weigths_file_name = weigths_file_name
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.batch_size = batch_size
        # from torchsummary import summary #network summary
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.token = "5823057823:AAE7Uo4nz2GduJVZYDoX_rPrEvmqYJmNUf0"
        self.chatIds = [168283555]
    
    def sendToTelegram(self, toPrint):
        for chat_id in self.chatIds:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage?chat_id={chat_id}&text={toPrint}"
            requests.get(url).json()
    
    def train(self, criterion, optimizer, scheduler, early_stopper,num_epochs=25,best_loss=100000.0, numTrain=1, acc_best_loss=0.0, alpha=0.12):
        train_losses=[]
        val_losses=[]
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = best_loss
        acc_best_loss = acc_best_loss

        toPrint = f'--------------------------------------\nCiao , sto per allenare {self.weigths_file_name}'
        self.sendToTelegram(toPrint)

        for epoch in range(num_epochs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for sample_batched in tqdm(self.dataloaders[phase]):
                    inputs = sample_batched['image'].float().to(self.device)
                    labels1 = sample_batched['label'][0].long().to(self.device)
                    labels2 = sample_batched['label'][1].long().to(self.device)
                    labels3 = sample_batched['label'][2].long().to(self.device)

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        #outputs = model(inputs, one_hot_input)
                        out1, out2, out3 = self.model(inputs)
                        # loss = criterion(outputs, labels)
                        prob1 = nn.Softmax(dim=1)(out1)
                        prob2 = nn.Softmax(dim=1)(out2)
                        prob3 = nn.Softmax(dim=1)(out3)

                        pred1 = torch.argmax(prob1,dim=1)
                        pred2 = torch.argmax(prob2,dim=1)
                        pred3 = torch.argmax(prob3,dim=1)

                        mask1 = torch.logical_and(torch.logical_and(torch.logical_and(torch.logical_and((labels2 != 3 ), ( labels2 != 4 )), ( labels2 != 5 )) , ( labels3 != 2 )), ( labels3 != 3))  
                        mask3 = torch.logical_and(torch.logical_and((labels2 != 3 ),  ( labels2 != 4 )),  ( labels2 != 5))
                        # Calculate loss for each sample using the masks
                        loss1 = criterion(out1[mask1].squeeze(), labels1[mask1].squeeze())
                        loss2 = criterion(out2, labels2)
                        loss3 = criterion(out3[mask3].squeeze(), labels3[mask3].squeeze())
                        task_loss = torch.stack([loss1,loss2,loss3])

                        starting_loss = torch.sum(task_loss)
                        
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            # compute the weighted loss w_i(t) * L_i(t)
                            weighted_task_loss = torch.mul(self.model.weights, task_loss)
                        
                            if epoch == 0:
                                # set L(0)
                                if torch.cuda.is_available():
                                    initial_task_loss = task_loss.data.cpu()
                                else:
                                    initial_task_loss = task_loss.data
                                initial_task_loss = initial_task_loss.numpy()
                                
                            # get the total loss
                            loss = torch.sum(weighted_task_loss)
                            
                            # clear the gradients
                            optimizer.zero_grad()
                            # do the backward pass to compute the gradients for the whole set of weights
                            # This is equivalent to compute each \nabla_W L_i(t)
                            loss.backward(retain_graph=True)

                            # set the gradients of w_i(t) to zero because these gradients have to be updated using the GradNorm loss
                            #print('Before turning to 0: {}'.format(self.model.weights.grad))
                            self.model.weights.grad.data = self.model.weights.grad.data * 0.0
                            #print('Turning to 0: {}'.format(self.model.weights.grad))
                                
                            # get layer of shared weights
                            W = self.model.get_last_shared_layer()

                            # get the gradient norms for each of the tasks
                            # G^{(i)}_w(t) 
                            norms = []
                            for i in range(len(task_loss)):
                                # get the gradient of this task loss with respect to the shared parameters
                                gygw = torch.autograd.grad(task_loss[i]+ 1e-15, W.parameters(), retain_graph=True)
                                # compute the norm
                                norms.append(torch.norm(torch.mul(self.model.weights[i], gygw[0])))
                            norms = torch.stack(norms)
                            #print('G_w(t): {}'.format(norms))


                            # compute the inverse training rate r_i(t) 
                            # \curl{L}_i 
                            if torch.cuda.is_available():
                                loss_ratio = task_loss.data.cpu().numpy() / (initial_task_loss + 1e-15)
                            else:
                                loss_ratio = task_loss.data.numpy() / (initial_task_loss + 1e-15)
                            # r_i(t)
                            inverse_train_rate = loss_ratio / (np.mean(loss_ratio) + 1e-15)
                            #print('r_i(t): {}'.format(inverse_train_rate))


                            # compute the mean norm \tilde{G}_w(t) 
                            if torch.cuda.is_available():
                                mean_norm = np.mean(norms.data.cpu().numpy())
                            else:
                                mean_norm = np.mean(norms.data.numpy())
                            #print('tilde G_w(t): {}'.format(mean_norm))


                            # compute the GradNorm loss 
                            # this term has to remain constant
                            constant_term = torch.tensor(mean_norm * (inverse_train_rate ** alpha), requires_grad=False)
                            if torch.cuda.is_available():
                                constant_term = constant_term.cuda()
                            #print('Constant term: {}'.format(constant_term))
                            # this is the GradNorm loss itself
                            grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
                            #print('GradNorm loss {}'.format(grad_norm_loss))

                            # compute the gradient for the weights
                            self.model.weights.grad = torch.autograd.grad(grad_norm_loss, self.model.weights)[0]

                            # do a step with the optimizer
                            optimizer.step()
                            # renormalize
                            normalize_coeff = 3 / torch.sum(self.model.weights.data, dim=0)
                            self.model.weights.data = self.model.weights.data * normalize_coeff

                    # statistics
                    running_loss += starting_loss.item()
                    pred1[~mask1] = labels1[~mask1]
                    pred3[~mask3] = labels3[~mask3]
                    correctness = (torch.logical_and(torch.logical_and(pred1 == labels1, pred2 == labels2), pred3 == labels3)).long()
                    running_corrects += sum(correctness)
                
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / (self.dataset_sizes[phase]/self.batch_size)
                epoch_acc = ((running_corrects / (self.dataset_sizes[phase]/self.batch_size))/self.batch_size)*100
                epoch_acc = epoch_acc.item()

                toPrint = f'{self.weigths_file_name}: Epochs {epoch}, {phase} Loss: {epoch_loss:.15f} Accuracy: {epoch_acc:.15f}'
                self.sendToTelegram(toPrint)

                if phase == 'val':
                    val_losses.append({'ValLoss': epoch_loss, 'Valacc': epoch_acc})
                    if early_stopper.early_stop(epoch_loss) == True:
                        time_elapsed = time.time() - since
                        self.sendToTelegram(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.\nStopped at epoch {epoch}')
                        self.sendToTelegram(f'Best val Acc: {best_loss:4f}')
                        self.sendToTelegram(f'Best val Acc: {acc_best_loss:4f}')

                        self.model.load_state_dict(best_model_wts)
                        torch.save(self.model.state_dict(), './baseline_models/'+ self.weigths_file_name+'.pt')
                        
                        with open('./baseline_evaluation_curves/'+self.weigths_file_name+ str(numTrain) +'training.json', 'w') as f:
                            dict = {'trainData' : train_losses,'valData' : val_losses, 'num epoch': epoch}
                            json.dump(dict, f)
                        
                        return self.model,best_loss,train_losses,val_losses 
                else:    
                    train_losses.append({'TrainLoss': epoch_loss, 'Valacc': epoch_acc})

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    acc_best_loss = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    torch.save(self.model.state_dict(), './baseline_models/'+ self.weigths_file_name+'.pt')
                    

        time_elapsed = time.time() - since
        
        
        toPrint = f'Training of {self.weigths_file_name} complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
        self.sendToTelegram(toPrint)
        toPrint = f'{self.weigths_file_name}: Best val loss: {best_loss:4f}, Best val Acc: {acc_best_loss:4f}'
        self.sendToTelegram(toPrint)

        

        with open('./baseline_evaluation_curves/'+self.weigths_file_name+ str(numTrain) +'training.json', 'w') as f:
            dict = {'trainData' : train_losses,'valData' : val_losses, 'num epoch': epoch}
            json.dump(dict, f)
        

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model,best_loss,train_losses,val_losses
    
    def train_norm_in_epoch(self, criterion, optimizer, scheduler, early_stopper,num_epochs=25,best_loss=100000.0, numTrain=1, acc_best_loss=0.0, alpha=0.12):
        train_losses=[]
        val_losses=[]
        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = best_loss
        acc_best_loss = acc_best_loss

        toPrint = f'--------------------------------------\nCiao , sto per allenare {self.weigths_file_name}'
        self.sendToTelegram(toPrint)

        for epoch in range(num_epochs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for sample_batched in tqdm(self.dataloaders[phase]):
                    inputs = sample_batched['image'].float().to(self.device)
                    labels1 = sample_batched['label'][0].long().to(self.device)
                    labels2 = sample_batched['label'][1].long().to(self.device)
                    labels3 = sample_batched['label'][2].long().to(self.device)

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        #outputs = model(inputs, one_hot_input)
                        out1, out2, out3 = self.model(inputs)
                        # loss = criterion(outputs, labels)
                        prob1 = nn.Softmax(dim=1)(out1)
                        prob2 = nn.Softmax(dim=1)(out2)
                        prob3 = nn.Softmax(dim=1)(out3)

                        pred1 = torch.argmax(prob1,dim=1)
                        pred2 = torch.argmax(prob2,dim=1)
                        pred3 = torch.argmax(prob3,dim=1)

                        mask1 = torch.logical_and(torch.logical_and(torch.logical_and(torch.logical_and((labels2 != 3 ), ( labels2 != 4 )), ( labels2 != 5 )) , ( labels3 != 2 )), ( labels3 != 3))  
                        mask3 = torch.logical_and(torch.logical_and((labels2 != 3 ),  ( labels2 != 4 )),  ( labels2 != 5))
                        # Calculate loss for each sample using the masks
                        loss1 = criterion(out1[mask1].squeeze(), labels1[mask1].squeeze())
                        loss2 = criterion(out2, labels2)
                        loss3 = criterion(out3[mask3].squeeze(), labels3[mask3].squeeze())
                        task_loss = torch.stack([loss1,loss2,loss3])

                        starting_loss = torch.sum(task_loss)
                        
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            # compute the weighted loss w_i(t) * L_i(t)
                            weighted_task_loss = torch.mul(self.model.weights, task_loss)
                        
                            if epoch == 0:
                                # set L(0)
                                if torch.cuda.is_available():
                                    initial_task_loss = task_loss.data.cpu()
                                else:
                                    initial_task_loss = task_loss.data
                                initial_task_loss = initial_task_loss.numpy()
                                
                            # get the total loss
                            loss = torch.sum(weighted_task_loss)
                            
                            # clear the gradients
                            optimizer.zero_grad()
                            # do the backward pass to compute the gradients for the whole set of weights
                            # This is equivalent to compute each \nabla_W L_i(t)
                            loss.backward(retain_graph=True)

                            # set the gradients of w_i(t) to zero because these gradients have to be updated using the GradNorm loss
                            #print('Before turning to 0: {}'.format(self.model.weights.grad))
                            self.model.weights.grad.data = self.model.weights.grad.data * 0.0
                            #print('Turning to 0: {}'.format(self.model.weights.grad))
                                
                            # get layer of shared weights
                            W = self.model.get_last_shared_layer()

                            # get the gradient norms for each of the tasks
                            # G^{(i)}_w(t) 
                            norms = []
                            for i in range(len(task_loss)):
                                # get the gradient of this task loss with respect to the shared parameters
                                gygw = torch.autograd.grad(task_loss[i]+ 1e-15, W.parameters(), retain_graph=True)
                                # compute the norm
                                norms.append(torch.norm(torch.mul(self.model.weights[i], gygw[0])))
                            norms = torch.stack(norms)
                            #print('G_w(t): {}'.format(norms))


                            # compute the inverse training rate r_i(t) 
                            # \curl{L}_i 
                            if torch.cuda.is_available():
                                loss_ratio = task_loss.data.cpu().numpy() / (initial_task_loss + 1e-15)
                            else:
                                loss_ratio = task_loss.data.numpy() / (initial_task_loss + 1e-15)
                            # r_i(t)
                            inverse_train_rate = loss_ratio / (np.mean(loss_ratio) + 1e-15)
                            #print('r_i(t): {}'.format(inverse_train_rate))


                            # compute the mean norm \tilde{G}_w(t) 
                            if torch.cuda.is_available():
                                mean_norm = np.mean(norms.data.cpu().numpy())
                            else:
                                mean_norm = np.mean(norms.data.numpy())
                            #print('tilde G_w(t): {}'.format(mean_norm))


                            # compute the GradNorm loss 
                            # this term has to remain constant
                            constant_term = torch.tensor(mean_norm * (inverse_train_rate ** alpha), requires_grad=False)
                            if torch.cuda.is_available():
                                constant_term = constant_term.cuda()
                            #print('Constant term: {}'.format(constant_term))
                            # this is the GradNorm loss itself
                            grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
                            #print('GradNorm loss {}'.format(grad_norm_loss))

                            # compute the gradient for the weights
                            self.model.weights.grad = torch.autograd.grad(grad_norm_loss, self.model.weights)[0]

                            # do a step with the optimizer
                            optimizer.step()

                    # statistics
                    running_loss += starting_loss.item()
                    pred1[~mask1] = labels1[~mask1]
                    pred3[~mask3] = labels3[~mask3]
                    correctness = (torch.logical_and(torch.logical_and(pred1 == labels1, pred2 == labels2), pred3 == labels3)).long()
                    running_corrects += sum(correctness)
                
                if phase == 'train':
                    print("self.model.weights.data pre normalization: ", self.model.weights.data)
                    # renormalize
                    normalize_coeff = 3 / torch.sum(self.model.weights.data, dim=0)
                    self.model.weights.data = self.model.weights.data * normalize_coeff
                    print("self.model.weights.data post normalization: ", self.model.weights.data)
                    scheduler.step()

                epoch_loss = running_loss / (self.dataset_sizes[phase]/self.batch_size)
                epoch_acc = ((running_corrects / (self.dataset_sizes[phase]/self.batch_size))/self.batch_size)*100
                epoch_acc = epoch_acc.item()

                toPrint = f'{self.weigths_file_name}: Epochs {epoch}, {phase} Loss: {epoch_loss:.15f} Accuracy: {epoch_acc:.15f}'
                self.sendToTelegram(toPrint)

                if phase == 'val':
                    val_losses.append({'ValLoss': epoch_loss, 'Valacc': epoch_acc})
                    if early_stopper.early_stop(epoch_loss) == True:
                        time_elapsed = time.time() - since
                        self.sendToTelegram(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.\nStopped at epoch {epoch}')
                        self.sendToTelegram(f'Best val Acc: {best_loss:4f}')
                        self.sendToTelegram(f'Best val Acc: {acc_best_loss:4f}')

                        self.model.load_state_dict(best_model_wts)
                        torch.save(self.model.state_dict(), './baseline_models/'+ self.weigths_file_name+'.pt')
                        
                        with open('./baseline_evaluation_curves/'+self.weigths_file_name+ str(numTrain) +'training.json', 'w') as f:
                            dict = {'trainData' : train_losses,'valData' : val_losses, 'num epoch': epoch}
                            json.dump(dict, f)
                        
                        return self.model,best_loss,train_losses,val_losses 
                else:    
                    train_losses.append({'TrainLoss': epoch_loss, 'Valacc': epoch_acc})

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    acc_best_loss = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    torch.save(self.model.state_dict(), './baseline_models/'+ self.weigths_file_name+'.pt')
                    

        time_elapsed = time.time() - since
        
        
        toPrint = f'Training of {self.weigths_file_name} complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
        self.sendToTelegram(toPrint)
        toPrint = f'{self.weigths_file_name}: Best val loss: {best_loss:4f}, Best val Acc: {acc_best_loss:4f}'
        self.sendToTelegram(toPrint)

        

        with open('./baseline_evaluation_curves/'+self.weigths_file_name+ str(numTrain) +'training.json', 'w') as f:
            dict = {'trainData' : train_losses,'valData' : val_losses, 'num epoch': epoch}
            json.dump(dict, f)
        

        # load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model,best_loss,train_losses,val_losses