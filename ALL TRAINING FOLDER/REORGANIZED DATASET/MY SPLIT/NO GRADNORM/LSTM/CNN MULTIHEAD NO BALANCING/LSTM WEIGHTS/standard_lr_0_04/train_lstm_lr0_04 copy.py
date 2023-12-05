import argparse
import cv2
from dataset import RSCDMultiLabel
from dataset import category_dict_sequential, category_dict_sequential_inv
import datetime
import json
from model import Encoder, Decoder
from model import convert_weights
from munkres import Munkres
import numpy as np
import os
import sys
from sklearn.metrics import precision_recall_fscore_support
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from torch.cuda import device_count, is_available
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm
import requests
# import debugpy
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

m = Munkres()
PARAMETERS_AND_NAME_MODEL = "train_lstm_backbone_b0_lr_0_04"
TOKEN = "5823057823:AAE7Uo4nz2GduJVZYDoX_rPrEvmqYJmNUf0"

chatIds = [168283555] #DA LASCIARE SOLTANTO IL MIO
#1407029395,163426269
# from torchsummary import summary #network summary
print(device_count())
device = torch.device("cuda" if is_available() else "cpu")
print(device)

def sendToTelegram(toPrint):
    for chat_id in chatIds:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={toPrint}"
        requests.get(url).json()
        
class SWA():
    """Average snapshots of a model to make the network generalize better."""
    def __init__(self, number_swa_models=0):
        """Init function."""
        self.number_swa_models = number_swa_models
        # super(SWA, self).__init__()

    def move_average(self, model, model_swa):
        """Change the weights of the SWA model."""
        self.number_swa_models += 1
        alpha = 1.0 / self.number_swa_models
        for param1, param2 in zip(model_swa.parameters(), model.parameters()):
            param1.data *= (1.0 - alpha)
            param1.data += param2.data * alpha

# def visualize_batch_fn(images, labels, label_lengths):
#     N = images.shape[0]
#     image_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3) 
#     image_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
#     for i in range(N):
#         image = images[i].data.cpu().numpy()
#         image = image.transpose(1, 2, 0)
#         image *= image_std
#         image += image_mean
#         image = (255.0 * image).astype(np.uint8)
#         indexes = labels[i].data.cpu().numpy().tolist()[1:label_lengths[i].item()-1]
#         indexes = [x for x in indexes]
#         labels_batch = [categories[x] for x in indexes]
#         cv2.imwrite("batches/%d.jpg" % i, image[:,:,::-1])
#         print('%d %s' % (i, ','.join(labels_batch)))
#     import epdb; epdb.set_trace()

def order_the_targets_mla(scores, targets, label_lengths_sorted):
    ###
    scores_tensor = scores.clone()
    targets_tensor = targets.clone()
    ###
    device = targets.device
    scores = scores.data.cpu().numpy()
    targets = targets.data.cpu().numpy()
    targets_new = targets.copy()
    N = scores.shape[0]
    time_steps = scores.shape[1]
    indexes = np.argmax(scores, axis=2)
    changed_batch_indexes = []
    for i in range(N):
        n_labels = label_lengths_sorted[i] - 1
        current_labels = targets_tensor[i][0:n_labels]
        cost_matrix = np.zeros((n_labels, n_labels), dtype=np.float32)
        for j in range(n_labels):
            losses = -F.log_softmax(scores_tensor[i][j], dim=0)
            temp = losses[current_labels]
            cost_matrix[j, :] = temp.data.cpu().numpy()
        indexes = m.compute(cost_matrix)
        new_labels = [x[1] for x in indexes]
        current_labels = current_labels.tolist()
        new_labels = [current_labels[x] for x in new_labels]
        targets_new[i][0:n_labels] = new_labels
    targets_new = torch.LongTensor(targets_new).to(device)
    return targets_new

def order_the_targets_pla(scores, targets, label_lengths_sorted):
    device = targets.device
    scores_tensor = scores.clone()
    scores = scores.data.cpu().numpy()
    targets = targets.data.cpu().numpy()
    targets_new = targets.copy()
    targets_newest = targets.copy()
    N = scores.shape[0]
    time_steps = scores.shape[1]
    indexes = np.argmax(scores, axis=2)
    changed_batch_indexes = []
    for i in range(N):
        common_indexes = set(targets[i][0:label_lengths_sorted[i]-1]).intersection(set(indexes[i]))
        diff_indexes = set(targets[i][0:label_lengths_sorted[i]-1]).difference(set(indexes[i]))
        diff_indexes_list = list(diff_indexes)
        common_indexes_copy = common_indexes.copy()
        index_array = np.zeros((len(diff_indexes), len(diff_indexes)))
        if common_indexes != set():
            changed_batch_indexes.append(i)
            for j in range(label_lengths_sorted[i] - 1):
                if indexes[i][j] in common_indexes:
                    if indexes[i][j] != targets_new[i][j].item():
                        old_value = targets_new[i][j]
                        new_value = indexes[i][j]
                        new_value_index = np.where(
                            targets_new[i] == new_value)[0][0]
                        targets_new[i][j] = new_value
                        targets_new[i][new_value_index] = old_value
                    common_indexes.remove(indexes[i][j].item())

        targets_newest[i] = targets_new[i]
        n_different = len(diff_indexes)
        if n_different > 1:
            diff_indexes_tuples = [[count, elem]
                                   for count, elem in enumerate(
                                           targets_new[i][0:label_lengths_sorted[i]-1])
                                   if elem in diff_indexes]
            diff_indexes_locations, diff_indexes_ordered = zip(
                *diff_indexes_tuples)
            cost_matrix = np.zeros((n_different, n_different),
                                   dtype=np.float32)
            for diff_count, diff_index_location in enumerate(
                    diff_indexes_locations):
                losses = -F.log_softmax(
                    scores_tensor[i][diff_index_location], dim=0)
                temp = losses[torch.LongTensor(diff_indexes_ordered)]
                cost_matrix[diff_count, :] = temp.data.cpu().numpy()
            indexes2 = m.compute(cost_matrix)
            new_labels = [x[1] for x in indexes2]
            for new_label_count, new_label in enumerate(new_labels):
                targets_newest[i][diff_indexes_locations[new_label_count]] = diff_indexes_ordered[new_label]

    targets_newest = torch.LongTensor(targets_newest).to(device)
    return targets_newest
    
def convert_to_array(scores, targets, target_lengths):
    scores = scores.data.cpu().numpy()
    targets = targets.data.cpu().numpy()
    number_class = 13
    N = scores.shape[0]
    preds = np.zeros((N, number_class), dtype=np.float32)
    labels = np.zeros((N, number_class), dtype=np.float32)
    number_time_steps = scores.shape[1]
    for i in range(N):
        preds_image = []
        for step_t in range(number_time_steps):
            step_pred = np.argmax(scores[i][step_t])
            if category_dict_sequential_inv[step_pred] == '<end>':
                break
            preds_image.append(step_pred)
        preds[i, preds_image] = 1
        labels_image = targets[i][0:target_lengths[i]-1]
        labels[i, labels_image] = 1
    return preds, labels

def my_collate(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch)

def adjust_learning_rate(optimizer, shrink_factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

#DEFINE MODEL
class MyClassifierModel(nn.Module):
        # x = self.Backbone.features[0](image)
        # x = self.Backbone.features[1](x)
        # x = self.Backbone.features[2](x)
        # x = self.Backbone.features[3](x)
        # x = self.Backbone.features[4](x)
        # x = self.Backbone.features[5](x)
        
        # x = self.Backbone.features[6](x)
        # x = self.Backbone.features[7](x)
        # x = self.Backbone.features[8](x)
        # feature = self.Backbone.avgpool(x)
        # ext_feature = torch.flatten(feature, 1)
        # out1 = self.Classifier_1(ext_feature)
        # out2 = self.Classifier_2(ext_feature)
        # out3 = self.Classifier_3(ext_feature)
        # return out1, out2, out3
    def __init__(self, Backbone, Classifier_1, Classifier_2, Classifier_3):
        super(MyClassifierModel, self).__init__()
        self.Backbone = Backbone
        self.Backbone.classifier = nn.Identity()
        self.Classifier_1 = Classifier_1
        self.Classifier_2 = Classifier_2
        self.Classifier_3 = Classifier_3
        
    def forward(self, image):
        return_tuple = []
        x = self.Backbone.features[0](image)
        x = self.Backbone.features[1](x)
        x = self.Backbone.features[2](x)
        x = self.Backbone.features[3](x)
        x = self.Backbone.features[4](x)
        x = self.Backbone.features[5](x)
        
        x = self.Backbone.features[6](x)
        x = self.Backbone.features[7](x)
        x = self.Backbone.features[8](x)
        return_tuple.append(x.permute(0, 2, 3, 1))
        feature = self.Backbone.avgpool(x)
        ext_feature = torch.flatten(feature, 1)
        return_tuple.append(ext_feature)
        # out1 = self.Classifier_1(ext_feature)
        # out2 = self.Classifier_2(ext_feature)
        # out3 = self.Classifier_3(ext_feature)
        return return_tuple

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-batch_size', default=128, type=int)
    parser.add_argument('-num_workers', default=3, type=int)
    parser.add_argument('-decoder_lr', default=0.04, type=float)
    parser.add_argument('-encoder_lr', default=0.01, type=float)
    parser.add_argument('-epochs', default=15, type=int)
    parser.add_argument('-snapshot', default=None)
    parser.add_argument('-hidden_size', default=512, type=int)
    parser.add_argument('-embed_size', default=256, type=int)
    parser.add_argument('-attention_size', default=512, type=int)
    parser.add_argument('-save_path', default=None)
    parser.add_argument('-test_model', action='store_true', default=False)
    parser.add_argument('-finetune_encoder', action='store_true', default=False)
    parser.add_argument('-visualize_batch', action='store_true', default=False)
    parser.add_argument('-order_free', type=str, default=None)
    parser.add_argument('-image_path',
                        help='Image path for the training and validation folders')
    parser.add_argument('-swa_params', type=str, default='{}')
    parser.add_argument('-train_from_scratch', action='store_true', default=False)
    parser.add_argument('-encoder_weights', default=None,
                        help='weights from the encoder training')
    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-sort_by_freq', action='store_true')
    parser.add_argument('-coeff', type=float, default=0.5)
    parser.add_argument('-epochs_to_decrease_lr', type=int, default=1)
    args = parser.parse_args()

    save_path = args.save_path
    print("Save path", save_path)
    test_model = args.test_model
    if not test_model:
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        log_path = os.path.join(save_path, 'logs')
        if not os.path.isdir(log_path):
            os.mkdir(log_path)
        else:
            if args.snapshot == None:
                raise ValueError('Delete the log path manually %s' % log_path)
        writer = SummaryWriter(log_dir=log_path)
    # ###DEBUG###
    # try:
    #     debugpy.listen(("localhost", 2517))
    #     debugpy.wait_for_client()
    # except:
    #     print("non mi fermo")
    #     pass
    finetune_encoder = args.finetune_encoder
    if finetune_encoder:
        print("FINETUNING THE ENCODER")
    else:
        print("NOT FINETUNING")
    if test_model is True:
        assert args.snapshot is not None
    else:
        if args.sort_by_freq is False:
            assert args.order_free in ["pla", "mla"]
        else:
            if args.order_free:
                raise ValueError('Sort by freq and order_free are mutually exclusive.')
    resume = 0
    highest_f1 = 0
    epochs_without_imp = 0
    iterations = 0
    # encoder = Encoder(encoder_weights=args.encoder_weights)

    # For a model pretrained on IMAGENET1K_V1
    backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    ClassifierModel_1 = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1280, 4),
        )
    ClassifierModel_2 = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1280, 6),
        )
    ClassifierModel_3 = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(1280, 5),
        )
    # #FINAL MODEL
    encoder = MyClassifierModel(backbone, ClassifierModel_1, ClassifierModel_2, ClassifierModel_3)
    #load weigths for this model
    checkpoint1 = torch.load(args.encoder_weights)
    encoder.load_state_dict(checkpoint1)
    
    for param in encoder.parameters(): 
        param.requires_grad = False
        
    decoder = Decoder(args.hidden_size, args.embed_size, args.attention_size, args.dropout)
    encoder = encoder.to('cuda')
    decoder = decoder.to('cuda')

    snapshot = args.snapshot
    test_model = args.test_model
    train_from_scratch = args.train_from_scratch
    swa_params = eval(args.swa_params)
    finetune_encoder = args.finetune_encoder

    if not test_model:
        if finetune_encoder:
            encoder_optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.encoder_lr)
        decoder_optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.decoder_lr)
    else:
        print("Testing the model")


    checkpoint = None
    if snapshot:
        checkpoint = torch.load(snapshot,  map_location=lambda storage, loc: storage)
        if (train_from_scratch and 'decoder_swa_state_dict' in checkpoint) or (test_model and 'decoder_swa_state_dict' in checkpoint):
            print("Inputting the swa weights.")
            decoder.load_state_dict(convert_weights(checkpoint['decoder_swa_state_dict']))
            if 'encoder_swa_state_dict' in checkpoint:
                encoder.load_state_dict(convert_weights(checkpoint['encoder_swa_state_dict']))
            else:
                encoder.load_state_dict(convert_weights(checkpoint['encoder_state_dict']))
        else:
            encoder.load_state_dict(convert_weights(checkpoint['encoder_state_dict']))
            decoder.load_state_dict(convert_weights(checkpoint['decoder_state_dict']))
        if args.test_model == False and args.train_from_scratch == False:
            resume = checkpoint['resume'] + 1
            highest_f1 = checkpoint['f1']
            iterations = checkpoint['iterations']
            epochs_without_imp = checkpoint['epochs_without_imp']
            if finetune_encoder:
                encoder_optimizer.load_state_dict(
                    checkpoint['encoder_optimizer_state_dict'])
            decoder_optimizer.load_state_dict(
                checkpoint['decoder_optimizer_state_dict'])

    if swa_params:
        from lr_scheduler import CyclicalLR
        swa_coeff = swa_params.get('swa_coeff', 0.1)
        if not args.test_model:
            scheduler_decoder = CyclicalLR(decoder_optimizer,
                                           swa_params['lr_high'],
                                           swa_params['lr_low'],
                                           swa_params['cycle_length'])
            if finetune_encoder:
                scheduler_encoder = CyclicalLR(encoder_optimizer,
                                               swa_params['lr_high'] * swa_coeff,
                                               swa_params['lr_low'] * swa_coeff,
                                               swa_params['cycle_length'])
        decoder_swa = Decoder(args.hidden_size, args.embed_size,
                              args.attention_size, args.dropout).to('cuda')
        encoder_swa = Encoder().to('cuda')
        print("Encoder and decoder learning rates will be overwritten")
        if checkpoint:
            decoder_swa.load_state_dict(convert_weights(checkpoint['decoder_swa_state_dict']))
            if 'encoder_swa_state_dict' in checkpoint:
                encoder_swa.load_state_dict(convert_weights(checkpoint['encoder_swa_state_dict']))
            else:
                raise ValueError("No encoder swa state dict")

            if args.train_from_scratch == False and args.test_model == False:
                iterations = checkpoint['iterations']
                number_swa_models = iterations / (swa_params['cycle_length'] + 1)
                print("# of SWA models", number_swa_models)

                swa = SWA(number_swa_models=number_swa_models)
                scheduler_decoder.curr_iter = iterations
                if finetune_encoder:
                    scheduler_encoder.curr_iter = iterations
                print(scheduler_decoder.get_lr()[0])
                print("SWA decoder curr lr", scheduler_decoder.print_lr()[0])
            else:
                swa = SWA(number_swa_models=0)
                print("# of SWA models 0")
        else:
            swa = SWA(number_swa_models=0)
            print("# of SWA models 0")

    encoder.eval()
    decoder.eval()
    if swa_params:
        encoder_swa.eval()
        decoder_swa.eval()

    criterion = nn.CrossEntropyLoss()

    train_dataset = RSCDMultiLabel(train=True,
                                   classification=False,
                                   image_path=args.image_path,
                                 sort_by_freq=args.sort_by_freq)
    test_dataset = RSCDMultiLabel(train=False,
                                  classification=False,
                                  image_path=args.image_path,
                                 sort_by_freq=args.sort_by_freq)
    dataloader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            shuffle=True,
                            drop_last=False,
                            collate_fn=my_collate)
    dataloader_val = DataLoader(test_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                shuffle=False,
                                drop_last=False,
                                collate_fn=my_collate)

    if torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
        if swa_params:
            encoder_swa = nn.DataParallel(encoder_swa)
            decoder_swa = nn.DataParallel(decoder_swa)

    best_f1 = 0.0
    train_losses=[]
    val_losses=[]
    for epoch in range(resume, args.epochs):
        training = True
        if args.test_model:
            training = False
        correct = 0
        n_batch = 0
        loss = torch.Tensor([0]).to('cuda')
        if training:
            # train
            if finetune_encoder:
                encoder.train()
                if swa_params:
                    encoder_swa.train()
            decoder.train()
            if swa_params:
                decoder_swa.train()

            for i, batch in enumerate(tqdm(dataloader)):
                iterations += 1
                images = batch[0]
                labels = batch[1]
                label_lengths = batch[2]
                labels_classification = batch[3].to('cuda')
                if args.visualize_batch:
                    visualize_batch_fn(images, labels, label_lengths)
                images = images.to('cuda')
                labels = labels.to('cuda')
                label_lengths = label_lengths.to('cuda')
                encoder_out, fc_out = encoder(images)
                if swa_params:
                    if finetune_encoder:
                        encoder_swa(images)
                scores, labels_sorted, label_lengths_sorted = decoder(
                    encoder_out, fc_out, labels, label_lengths)

                # multi-gpu support
                label_lengths_sorted, sort_ind = label_lengths_sorted.sort(dim=0, descending=True)
                labels_sorted = labels_sorted[sort_ind]
                scores = scores[sort_ind]

                # Since we decoded starting with <start>,
                # the targets are all words after <start>, up to <end>
                targets = labels_sorted[:, 1:]

                global_iter = epoch * len(dataloader) + i

                
                # training accuracy
                preds_train, labels_train = convert_to_array(scores, targets,
                                                                label_lengths_sorted)
                    # _, _, f1, _ = precision_recall_fscore_support(preds_train,
                    #                                               labels_train,
                    #                                               average='micro')
                    # writer.add_scalar('train_f1', 100 * f1, global_iter)
                correct += sum(((preds_train == labels_train).all(1)))
                n_batch += 1
                if args.order_free == 'pla':
                    # change the targets
                    targets = order_the_targets_pla(
                        scores, targets, label_lengths_sorted)
                elif args.order_free == 'mla':
                    targets = order_the_targets_mla(
                        scores, targets, label_lengths_sorted)   

                scores = pack_padded_sequence(
                    scores, label_lengths_sorted.cpu().numpy(), batch_first=True)
                targets = pack_padded_sequence(
                    targets, label_lengths_sorted.cpu().numpy(), batch_first=True)

                # Calculate loss
                loss_lstm = criterion(scores.data, targets.data)
                loss += loss_lstm
                # if i % 50 == 0:
                #     writer.add_scalar('loss', loss_lstm.item(), global_iter)
                #     # learning rates
                #     writer.add_scalar('decoder_lr', decoder_optimizer.param_groups[0]['lr'], global_iter)
                    # if finetune_encoder:
                        # writer.add_scalar('encoder_lr', encoder_optimizer.param_groups[0]['lr'], global_iter)
                decoder_optimizer.zero_grad()
                if finetune_encoder:
                    encoder_optimizer.zero_grad()
                loss_lstm.backward()
                decoder_optimizer.step()
                if finetune_encoder:
                    encoder_optimizer.step()

                if swa_params:
                    if iterations % (scheduler_decoder.cycle_length + 1) == 0:
                        swa.move_average(decoder, decoder_swa)
                        if finetune_encoder:
                            swa.move_average(encoder, encoder_swa)
                        if scheduler_decoder.print_lr()[0] != scheduler_decoder.lr_low:
                            raise AssertionError("""The learning rate is not at the lowest point.""")
                    scheduler_decoder.step()
                    if finetune_encoder:
                        scheduler_encoder.step()

                # if i % 50 == 0:
                #     print("epoch: %d/%d, batch: %d/%d ,loss: %.2f" % (
                #         epoch, args.epochs, i, len(dataloader), loss_lstm.item()))
            epoch_acc = correct/(n_batch*args.batch_size)*100
            epoch_loss = loss.item()/n_batch
            toPrint = f'{PARAMETERS_AND_NAME_MODEL}: Epochs {epoch}, train Loss: {epoch_loss:.15f} Accuracy: {epoch_acc:.15f}'
            sendToTelegram(toPrint)
            train_losses.append({'TrainLoss': epoch_loss, 'TrainAcc': epoch_acc})
        correct = 0
        n_batch = 0
        loss = torch.Tensor([0]).to('cuda')
        with torch.no_grad():
            # validation
            encoder.eval()
            decoder.eval()
            if swa_params:
                encoder_swa.eval()
                decoder_swa.eval()
            preds_all = None
            labels_all = None
            for i, batch in enumerate(tqdm(dataloader_val,
                                           total=len(dataloader_val))):
                images = batch[0]
                labels = batch[1]
                label_lengths = batch[2]
                images = images.to('cuda')
                labels = labels.to('cuda')
                label_lengths = label_lengths.to('cuda')

                if swa_params:
                    encoder_dict, fc_out = encoder_swa(images)
                    scores, labels_sorted, label_lengths_sorted = decoder_swa(
                        encoder_dict, fc_out, labels, label_lengths)
                    targets = labels_sorted[:, 1:]
                    preds, labels = convert_to_array(scores, targets,
                                                     label_lengths_sorted)

                else:
                    encoder_out, fc_out = encoder(images)
                    scores, labels_sorted, label_lengths_sorted = decoder(
                        encoder_out, fc_out, labels, label_lengths)
                    targets = labels_sorted[:, 1:]
                    preds_val, labels_val = convert_to_array(scores, targets,
                                                     label_lengths_sorted)
                    scores = pack_padded_sequence(
                        scores, label_lengths_sorted.cpu().numpy(), batch_first=True)
                    targets = pack_padded_sequence(
                        targets, label_lengths_sorted.cpu().numpy(), batch_first=True)

                    # Calculate loss
                    loss_lstm = criterion(scores.data, targets.data)
                    loss += loss_lstm.item()
                    correct += sum(((preds_val == labels_val).all(1)))
                    n_batch += 1
                if i == 0:
                    preds_all = preds_val
                    labels_all = labels_val
                else:
                    preds_all = np.concatenate((preds_all, preds_val), axis=0)
                    labels_all = np.concatenate((labels_all, labels_val), axis=0)
            epoch_acc = correct/(n_batch*args.batch_size)*100
            epoch_loss = loss.item()/n_batch
            toPrint = f'{PARAMETERS_AND_NAME_MODEL}: Epochs {epoch}, val Loss: {epoch_loss:.15f} Accuracy: {epoch_acc:.15f}'
            sendToTelegram(toPrint)
            val_losses.append({'ValLoss': epoch_loss, 'Valacc': epoch_acc})

        # this function mixes the precision and recall
        prec, recall, _, _ = precision_recall_fscore_support(preds_all,
                                                             labels_all,
                                                             average='macro')
        macro_f1 = 2 * prec * recall / (prec + recall)
        print("MACRO prec %.2f%%, recall %.2f%%, f1 %.2f%%" % (
            recall * 100, prec * 100, macro_f1 * 100))

        prec, recall, f1, _ = precision_recall_fscore_support(preds_all,
                                                              labels_all,
                                                              average='micro')
        print("MICRO prec %.2f%%, recall %.2f%%, f1 %.2f%%" % (
            recall * 100, prec * 100, f1 * 100))

        if args.test_model:
            break
        else:
            writer.add_scalar('micro_f1', f1 * 100, epoch)
            writer.add_scalar('macro_f1', macro_f1 * 100, epoch)
            save_dict = {'encoder_state_dict': encoder.state_dict(),
                         'decoder_state_dict': decoder.state_dict(),
                         'resume': epoch, 'f1': f1, 'iterations': iterations,
                         'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                         'epochs_without_imp': epochs_without_imp}
            if swa_params:
                save_dict['decoder_swa_state_dict'] = decoder_swa.state_dict()
            if finetune_encoder:
                save_dict['encoder_optimizer_state_dict'] = encoder_optimizer.state_dict()
                if swa_params:
                    save_dict['encoder_swa_state_dict'] = encoder_swa.state_dict()
            torch.save(save_dict, save_path + '/checkpoint.pth.tar')
            if f1 > highest_f1:
                print("Highest f1 score was %.2f%% now it is %.2f%%" % (highest_f1*100.0, f1*100.0))
                highest_f1 = f1
                torch.save(save_dict, save_path + "/BEST_checkpoint.pth.tar")
                epochs_without_imp = 0
            else:
                epochs_without_imp += 1
                print("Highest f1 score is still %.2f%%, epochs without imp. %d" % (
                    highest_f1*100, epochs_without_imp))
                if epochs_without_imp == args.epochs_to_decrease_lr and swa_params == {}:
                    adjust_learning_rate(decoder_optimizer, args.coeff)
                    if finetune_encoder:
                        adjust_learning_rate(encoder_optimizer, args.coeff)
                    epochs_without_imp = 0
    with open(save_path+'/baseline_evaluation_curves/'+PARAMETERS_AND_NAME_MODEL +'training.json', 'w') as f:
        dict = {'trainData' : train_losses,'valData' : val_losses, 'num epoch': epoch}
        json.dump(dict, f)
