from model import Decoder
from dataset import category_dict_sequential,category_dict_sequential_inv
from torch.cuda import device_count, is_available
from torch import device, load, no_grad, sort, ones, flatten
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import numpy as np
import torch.nn as nn
import torch
from torchvision import transforms
import difflib
# import debugpy
import gradio as gr
from create_crop import crop_center
import gradio as gr
from torch.cuda import device_count, is_available
from torch import device, set_grad_enabled, argmax, sum, save, is_tensor, Tensor, load, no_grad, logical_or, ones
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torch.nn as nn
from torchvision import transforms
import torch
import numpy as np

#INSERT_IF_YOU_WANT_TO_OBTAIN_PERFORMANCES_ON_TEST_OR_ON_VAL VALUES:
#test
#validation
WEIGHT_PATH_LSTM = "/home/mattia/Desktop/Tesi/demo/models_weight/weight_lstm_model.tar"
WEIGHT_PATH_Base_model = "/home/mattia/Desktop/Tesi/demo/models_weight/Paper_baseline_new_split.pt"

# Imposta il dispositivo (cuda se disponibile, altrimenti cpu)
device = device("cuda" if is_available() else "cpu")

def find_most_similar_string(input_string, classes):
    # Suddividi la stringa in sottostringhe
    input_substrings = input_string.split('-')

    # Inizializza le variabili per tenere traccia del massimo numero di sottostringhe corrispondenti
    max_matches = 0
    most_similar_index = -1

    for i, class_string in enumerate(classes):
        # Suddividi la stringa dalla lista classes in sottostringhe
        class_substrings = class_string.split('-')

        # Conta il numero di sottostringhe in comune tra le due liste
        matches = len(set(input_substrings) & set(class_substrings))

        # Aggiorna se trovi un nuovo massimo
        if matches > max_matches:
            max_matches = matches
            most_similar_index = i

    # Se non hai trovato alcuna corrispondenza, utilizza difflib per trovare la stringa più simile
    if most_similar_index == -1:
        most_similar = difflib.get_close_matches(input_string, classes, n=1)
        if most_similar:
            most_similar_index = classes.index(most_similar[0])

    return most_similar_index

def format_list(tensor, string_list, classes):
    # Creare la lista risultante
    result_list = []

    # Creare un dizionario di corrispondenze tra stringhe e indici in classes
    class_index_map = {string: index for index, string in enumerate(classes)}

    # Iterare attraverso il tensore e creare la lista risultante con controllo
    for row in tensor:
        row_strings = ""
        index=0
        for i in row:
            if i < len(string_list):
                if index == 0:
                    tmp = string_list[i] + "-"
                else:
                    row_strings += string_list[i] + "-"
            else:
                if index < 3:
                    row_strings = tmp + row_strings
                    break                  
                else:
                    row_strings += tmp
                    break
            index += 1
        
        result_list.append(row_strings[:-1])
    # Trova l'indice della stringa più simile in classes
    to_ret = [find_most_similar_string(stringa, classes) for stringa in result_list]

    return result_list, to_ret

def convert_to_array(scores):
    scores = scores.data.cpu().numpy()
    number_class = 13
    N = scores.shape[0]
    preds = np.zeros((N, number_class), dtype=np.float32)
    number_time_steps = scores.shape[1]
    for i in range(N):
        preds_image = []
        for step_t in range(number_time_steps):
            step_pred = np.argmax(scores[i][step_t])
            if category_dict_sequential_inv[step_pred] == '<end>':
                break
            preds_image.append(step_pred)
        preds[i, preds_image] = 1
    return preds

# ###DEBUG###
# try:
#     debugpy.listen(("localhost", 2517))
#     debugpy.wait_for_client()
# except:
#     print("non mi fermo")
#     pass

#DEFINE MODEL
class MyClassifierModel_lstm(nn.Module):
    def __init__(self, Backbone, Classifier_1, Classifier_2, Classifier_3):
        super(MyClassifierModel_lstm, self).__init__()
        self.Backbone = Backbone
        self.Backbone.classifier = nn.Identity()
        self.Classifier_1 = Classifier_1
        self.Classifier_2 = Classifier_2
        self.Classifier_3 = Classifier_3
        self.weights = nn.Parameter(ones(3).float())
        
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
        ext_feature = flatten(feature, 1)
        return_tuple.append(ext_feature)
        # out1 = self.Classifier_1(ext_feature)
        # out2 = self.Classifier_2(ext_feature)
        # out3 = self.Classifier_3(ext_feature)
        return return_tuple

class MyClassifierModel_base_model(nn.Module):
    def __init__(self, Backbone, Classifier):
        super(MyClassifierModel_base_model, self).__init__()
        self.Backbone = Backbone
        self.Backbone.classifier = Classifier
        
    def forward(self, image):
        out = self.Backbone(image)
        return out
    
checkpoint = load(WEIGHT_PATH_LSTM)

###LSTM_MODEL_PART###
# For a model pretrained on IMAGENET1K_V1
backbone_lstm = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

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
encoder = MyClassifierModel_lstm(backbone_lstm, ClassifierModel_1, ClassifierModel_2, ClassifierModel_3)
decoder = Decoder(512, 256, 512, 0.0)

encoder.load_state_dict(checkpoint["encoder_state_dict"])
decoder.load_state_dict(checkpoint["decoder_state_dict"])
encoder = encoder.to('cuda')
decoder = decoder.to('cuda')
encoder.eval()
decoder.eval()

###BASE_MODEL_PART###
# For a model pretrained on VGGFace2
backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

ClassifierModel = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(1280, 27),
)

# Carica il modello allenato
model = MyClassifierModel_base_model(backbone, ClassifierModel)
model.load_state_dict(load(WEIGHT_PATH_Base_model))
for param in model.parameters(): 
    param.requires_grad = False
model.to(device)
model.eval()

# Definisci la trasformazione per il test
data_transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.498, 0.498, 0.498], [0.500, 0.500, 0.500]),
])

RSCD_categories = ['smooth', 'slight', 'severe', 'dry', 'wet',
              'water', 'fresh_snow', 'melted_snow', 'ice', 'asphalt', 'concrete',
              'gravel', 'mud']

classes = ["ice",  "fresh_snow",  "melted_snow",  "wet-mud",  "wet-gravel",  "wet-asphalt-slight",
           "wet-asphalt-smooth",  "wet-asphalt-severe",  "wet-concrete-slight",  "wet-concrete-smooth",
           "wet-concrete-severe",  "water-mud",  "water-gravel",  "water-asphalt-slight",  "water-asphalt-smooth",
           "water-asphalt-severe",  "water-concrete-slight",  "water-concrete-smooth",  "water-concrete-severe",
           "dry-mud",  "dry-gravel",  "dry-asphalt-slight",  "dry-asphalt-smooth",  "dry-asphalt-severe",
           "dry-concrete-slight",  "dry-concrete-smooth",  "dry-concrete-severe"]

class_dictionary = {
    "0":"3-5-4",
    "1":"3-3-4",
    "2":"3-4-4",
    "3":"3-1-2",
    "4":"3-1-3",
    "5":"1-1-0",
    "6":"0-1-0",
    "7":"2-1-0",
    "8":"1-1-1",
    "9":"0-1-1",
    "10":"2-1-1",
    "11":"3-2-2",
    "12":"3-2-3",
    "13":"1-2-0",
    "14":"0-2-0",
    "15":"2-2-0",
    "16":"1-2-1",
    "17":"0-2-1",
    "18":"2-2-1",
    "19":"3-0-2",
    "20":"3-0-3",
    "21":"1-0-0",
    "22":"0-0-0",
    "23":"2-0-0",
    "24":"1-0-1",
    "25":"0-0-1",
    "26":"2-0-1",
}

def crop_image(input_image):
    return crop_center(input_image)
    
def process_single_image_lstm(input_image):
    with no_grad():
        input_image = crop_center(input_image)
        input = data_transforms_test(input_image).float().to(device)
        input = torch.unsqueeze(input, 0)
        
        encoder_out, fc_out = encoder(input)
        scores = decoder(
            encoder_out, fc_out)
        preds_train = convert_to_array(scores)
        indices = np.where(preds_train == 1)
        final_prediction = ''.join(category_dict_sequential_inv[indices[1][i]]+'-' for i in range(len(indices[1])))[:-1]
        return final_prediction
        
def process_single_image_base(input_image):
    with no_grad():
        input_image = crop_center(input_image)
        input = data_transforms_test(input_image).float().to(device)
        input = torch.unsqueeze(input, 0)
        #outputs = model(inputs, one_hot_input)
        out = model(input)
        # loss = criterion(outputs, labels)
        prob = nn.Softmax(dim=1)(out)

        pred = argmax(prob,dim=1)

        predict = classes[pred]
        return predict


# Interfaccia per il ritaglio
crop_interface = gr.Interface(
    fn=crop_image,
    inputs=gr.Image(),
    outputs=gr.Image(),
)

# Create an interface with a custom label for the output
process_interface_lstm = gr.Interface(
    fn=process_single_image_lstm,
    inputs=gr.Image(),
    outputs=gr.Textbox(label="Output Label for LSTM"),
)

# Create another interface with a custom label for the output
process_interface_base = gr.Interface(
    fn=process_single_image_base,
    inputs=gr.Image(),
    outputs=gr.Textbox(label="Output Label for Base"),
)

# Combine both interfaces into a single page
gr.mix.Parallel(crop_interface, process_interface_lstm, process_interface_base).launch()
