import os
import torch
from tqdm import tqdm
from exp.get_text_feature import get_text_features_path, get_text_encoder_dir
class TextTensorDataset(torch.utils.data.Dataset):
    def __init__(self, input_tensor, label_tensor, eot_indices):
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        self.eot_indices = eot_indices
    
    def __getitem__(self, index):
        return self.input_tensor[index], self.label_tensor[index], self.eot_indices[index]

    def __len__(self):
        return self.input_tensor.size(0)



def get_text_dataset_per_class(text_dataset):
    print("Building text dataset per class...")
    text_dataset_per_class = {}
    for text, text_label, eot_indices in tqdm(text_dataset):
        text_label = int(text_label)
        if text_label not in text_dataset_per_class:
            text_dataset_per_class[text_label] = []
        text_dataset_per_class[text_label].append([text, eot_indices])
    num_of_templates = len(text_dataset_per_class[text_label])
    for text_label in text_dataset_per_class:
        assert len(text_dataset_per_class[text_label]) == num_of_templates
    return text_dataset_per_class, num_of_templates

def get_zero_shot_weights(text_dataset, num_classes, in_features, text_encoder, device="cuda"):
    with torch.no_grad():
        text_dataset_per_class, _ = get_text_dataset_per_class(text_dataset)
        weights = torch.zeros(num_classes, in_features)
        for label in range(num_classes):
            texts = None
            eot_indices = None
            for i in range(len(text_dataset_per_class[label])):
                text, eot_indice = text_dataset_per_class[label][i]
                text = text.unsqueeze(0).to(device)  #1*512
                eot_indice = eot_indice.unsqueeze(0).to(device)  #1
                if texts is None:
                    texts = text
                    eot_indices = eot_indice
                else:
                    texts = torch.cat([texts, text], dim=0)
                    eot_indices = torch.cat([eot_indices, eot_indice], dim=0)
            text_features = text_encoder(texts, eot_indices)  
            text_features = text_features.mean(dim=0) 
            weights[label] = text_features
        # normalize the weights
        weights.data = torch.nn.functional.normalize(weights, dim=1)
    return weights


def get_512_linear_head_weight(dataset_name, n_cls):

    text_features_path = get_text_features_path(
            dataset_name, # args.dataset,
            "path_to_store_get_features", #args.feature_dir,
            "ViT-B-16", # args.clip_encoder
            "0", #args.text_layer_idx
            "hand_crafted" # args.text_augmentation
        )
    text_features = torch.load(text_features_path)
    text_dataset = TextTensorDataset(
        text_features['features'],  # n_cls*512
        text_features['labels'],   # n_cls
        text_features['eot_indices'])  # n_cls


    text_encoder_dir = get_text_encoder_dir(
        "path_to_store_get_features", #args.feature_dir,
        "ViT-B-16", # args.clip_encoder
        "0", #args.text_layer_idx
    )
    text_encoder_path = os.path.join(text_encoder_dir, "encoder.pth")
    text_encoder = torch.load(text_encoder_path).partial_model.train().cuda()

    weight_data = get_zero_shot_weights(text_dataset, n_cls, 512, text_encoder)
    return weight_data

def get_1024_linear_head_weight(dataset_name, n_cls):

    text_features_path = get_text_features_path(
            dataset_name, # args.dataset,
            "path_to_store_get_features", #args.feature_dir,
            "ViT-B-16", # args.clip_encoder
            "0", #args.text_layer_idx
            "hand_crafted" # args.text_augmentation
        )
    text_features = torch.load(text_features_path)
    text_dataset = TextTensorDataset(
        text_features['features'],  # n_cls*512
        text_features['labels'],   # n_cls
        text_features['eot_indices'])  # n_cls


    text_encoder_dir = get_text_encoder_dir(
        "path_to_store_get_features", #args.feature_dir,
        "ViT-B-16", # args.clip_encoder
        "0", #args.text_layer_idx
    )
    text_encoder_path = os.path.join(text_encoder_dir, "encoder.pth")
    text_encoder = torch.load(text_encoder_path).partial_model.train().cuda()

    # [M,512]
    weight_data = get_zero_shot_weights(text_dataset, n_cls, 512, text_encoder)
    # [M,1024]
    weight_data_1024 = torch.cat((weight_data , weight_data ), dim=1)
    return weight_data_1024

def get_no_norm_zero_shot_weights(text_dataset, num_classes, in_features, text_encoder, device="cuda"):
    with torch.no_grad():
        text_dataset_per_class, _ = get_text_dataset_per_class(text_dataset)
        weights = torch.zeros(num_classes, in_features)
        for label in range(num_classes):
            texts = None
            eot_indices = None
            for i in range(len(text_dataset_per_class[label])):
                text, eot_indice = text_dataset_per_class[label][i]
                text = text.unsqueeze(0).to(device)  #1*512
                eot_indice = eot_indice.unsqueeze(0).to(device)  #1
                if texts is None:
                    texts = text
                    eot_indices = eot_indice
                else:
                    texts = torch.cat([texts, text], dim=0)
                    eot_indices = torch.cat([eot_indices, eot_indice], dim=0)
            text_features = text_encoder(texts, eot_indices)  
            text_features = text_features.mean(dim=0)  
            weights[label] = text_features
        # not normalize!
        # weights.data = torch.nn.functional.normalize(weights, dim=1)
    return weights

def get_no_norm_1024_linear_head_weight(dataset_name, n_cls):

    text_features_path = get_text_features_path(
            dataset_name, # args.dataset,
            "path_to_store_get_features", #args.feature_dir,
            "ViT-B-16", # args.clip_encoder
            "0", #args.text_layer_idx
            "hand_crafted" # args.text_augmentation
        )
    text_features = torch.load(text_features_path)
    text_dataset = TextTensorDataset(
        text_features['features'],  # n_cls*512
        text_features['labels'],   # n_cls
        text_features['eot_indices'])  # n_cls


    text_encoder_dir = get_text_encoder_dir(
        "path_to_store_get_features", #args.feature_dir,
        "ViT-B-16", # args.clip_encoder
        "0", #args.text_layer_idx
    )
    text_encoder_path = os.path.join(text_encoder_dir, "encoder.pth")
    text_encoder = torch.load(text_encoder_path).partial_model.train().cuda()

    # [M,512]
    weight_data = get_no_norm_zero_shot_weights(text_dataset, n_cls, 512, text_encoder)
    # [M,1024]
    weight_data_1024 = torch.cat((weight_data , weight_data ), dim=1)
    return weight_data_1024


def get_no_norm_512_linear_head_weight(dataset_name, n_cls):

    text_features_path = get_text_features_path(
            dataset_name, # args.dataset,
            "path_to_store_get_features", #args.feature_dir,
            "ViT-B-16", # args.clip_encoder
            "0", #args.text_layer_idx
            "hand_crafted" # args.text_augmentation
        )
    text_features = torch.load(text_features_path)
    text_dataset = TextTensorDataset(
        text_features['features'],  # n_cls*512
        text_features['labels'],   # n_cls
        text_features['eot_indices'])  # n_cls


    text_encoder_dir = get_text_encoder_dir(
        "path_to_store_get_features", #args.feature_dir,
        "ViT-B-16", # args.clip_encoder
        "0", #args.text_layer_idx
    )
    text_encoder_path = os.path.join(text_encoder_dir, "encoder.pth")
    text_encoder = torch.load(text_encoder_path).partial_model.train().cuda()

    # [M,512]
    weight_data = get_no_norm_zero_shot_weights(text_dataset, n_cls, 512, text_encoder)
    return weight_data  

def get_text_dataset(dataset_name):

    text_features_path = get_text_features_path(
            dataset_name, # args.dataset,
            "path_to_store_get_features", #args.feature_dir,
            "ViT-B-16", # args.clip_encoder
            "0", #args.text_layer_idx
            "hand_crafted" # args.text_augmentation
        )
    text_features = torch.load(text_features_path)
    text_dataset = TextTensorDataset(
        text_features['features'],  # n_cls*512
        text_features['labels'],   # n_cls
        text_features['eot_indices'])  # n_cls

    return text_dataset
