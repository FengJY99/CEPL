import os
import os.path as osp
from collections import OrderedDict
import math
from tqdm import tqdm
import time
import datetime

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from get_linear_head_weight import get_512_linear_head_weight
from get_linear_head_weight import get_no_norm_512_linear_head_weight
from adaptation.lora import lora_replace_attention_layers, lora_replace_linear_layers

from clip import clip, model
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


import numpy as np

try:
    from scipy.ndimage import binary_erosion
except ImportError:
    pass  # Don't fail if scipy is not installed. It's only necessary for this one file.


_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = { 
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.", 
}
DATASET_NAME = { 
    "OxfordPets": "oxford_pets",
    "OxfordFlowers": "oxford_flowers",
    "FGVCAircraft": "fgvc_aircraft",
    "DescribableTextures": "dtd",
    "EuroSAT": "eurosat",
    "StanfordCars": "stanford_cars",
    "Food101": "food101",
    "SUN397": "sun397",
    "Caltech101": "caltech101",
    "UCF101": "ucf101",
    "ImageNet": "imagenet",  
    "ImageNetSketch": "imagenet_sketch",
    "ImageNetV2": "imagenetv2",
    "ImageNetA": "imagenet_a",
    "ImageNetR": "imagenet_r",
}



def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = lora_replace_linear_layers(clip_model.transformer,lora_r=4,lora_alpha=1,lora_dropout=0,start_block=3)
        self.transformer = lora_replace_attention_layers(self.transformer, lora_r=4, lora_alpha=1, lora_dropout=0, start_block=3)  
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    

class TransBlock(nn.Module):
    def __init__(self, width, heads, scale):
        super().__init__()
        self.atten = model.ResidualAttentionBlock(width, heads)
        self.ln_post = model.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, 512))  

    def Pruning(self, im_features, keep_rate=0.5):
        scale = im_features.size(2) ** -0.5 
        left_tokens = math.ceil(keep_rate * (im_features.size(1) - 1))+1
        cls_attn = (im_features[:, 0, :].unsqueeze(1) @ im_features.transpose(-2, -1)) * scale#[B, 1, C] [B, C, N] -->[B, 1, N]
        cls_attn = cls_attn.squeeze(1)  
        _, idx = torch.topk(cls_attn[:, 0:], left_tokens, dim=1, largest=True, sorted=True)  # [B, left_tokens]
        index = idx.unsqueeze(-1).expand(-1, -1, im_features.size(2))  # [B, left_tokens, C]  
        feat_output = torch.gather(im_features, dim=1, index=index)  # [B, left_tokens, C]
        feat_output = torch.mean(feat_output, dim=1, keepdim=True) #[B, 1, C]
        feat_output = feat_output.squeeze(1)  # [B, C]
        return feat_output # [B, C]

    def forward(self, x):
        x = self.atten(x)
        x = x.permute(1, 0, 2)
        x =self.Pruning(x) 
        x = self.ln_post(x) # [B, C]
        x = x @ self.proj # [B, C] [C, D] -> [B, D]
        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.EFF_PROMPTS.N_CTX
        ctx_init = cfg.TRAINER.EFF_PROMPTS.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # random initialization
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype) #4*512
        nn.init.normal_(ctx_vectors, std=0.02)
        prompts = " ".join(["X"] * n_ctx)
        prompts = prompts + "."
        print(f'Initial context: "{prompts}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized     

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))
        if cfg.TRAINER.EFF_PROMPTS.PREC == "fp16":
            self.meta_net.half()

        tokenized_prompts = clip.tokenize(prompts) # 1*77
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)  

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def construct_prompts(self, ctx, prefix, suffix):
        prompts = torch.cat(
            [
                prefix,  # (1, 1, dim)
                ctx,     # (1, n_ctx, dim)
                suffix,  # (1, *, dim)
            ],
            dim=1,
        )

        return prompts 
    

    def forward(self, im_features):
        prefix = self.token_prefix  
        suffix = self.token_suffix  

        ctx = self.ctx                     # (n_ctx, ctx_dim) 
        bias = self.meta_net(im_features)  # (batch, ctx_dim) 
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim) 
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim) 
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)


        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:  
            ctx_i = ctx_shifted_i.unsqueeze(0)  
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  
            prompts.append(pts_i)

        prompts = torch.cat(prompts, dim=0)  

        return prompts 


class LogitHead(nn.Module):
    def __init__(self, head, logit_scale=float(np.log(1 / 0.07))):
        super().__init__()
        self.head = head
        self.logit_scale = logit_scale
        self.in_features = head.in_features
        # Not learnable for simplicity
        self.logit_scale = torch.FloatTensor([logit_scale]).cuda()
        # Learnable
        # self.logit_scale = torch.nn.Parameter(torch.ones([]) * logit_scale)

    def forward(self, x):
        x = F.normalize(x, dim=1) 
        x = self.head(x)
        x = x * self.logit_scale.exp()
        return x
    

class CustomCLIP(nn.Module):
    def __init__(self, cfg, dm, classnames, clip_model):
        super().__init__()
        self.dm = dm
        self.dataset = cfg.DATASET.NAME
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.image_encoder.transformer = lora_replace_linear_layers(self.image_encoder.transformer,lora_r=4,lora_alpha=1,lora_dropout=0,start_block=3)
        self.image_encoder.transformer = lora_replace_attention_layers(self.image_encoder.transformer, lora_r=4, lora_alpha=1, lora_dropout=0, start_block=3) 
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.n_cls = self.prompt_learner.n_cls
        self.clip_model_token_embedding = clip_model.token_embedding.weight
        self.alpha = 1

        self.token_embedding_weight = self.clip_model_token_embedding.type(self.dtype)  
        self.embedding_weight_t = torch.nn.Parameter(torch.ones([]) * self.token_embedding_weight.t())

        state_dict = clip_model.state_dict()
        width = state_dict["visual.conv1.weight"].shape[0]     
        heads = state_dict["ln_final.weight"].shape[0] // 64   
        scale = width ** -0.5       
        self.trans_block = TransBlock(width, heads, scale)

        dataset_name = DATASET_NAME[self.dataset]  
        if (dataset_name=="imagenet") or (dataset_name =="sun397"):
            logit_head = nn.Linear(512, self.n_cls, bias=False)  
            self.norm_text_features = get_512_linear_head_weight(dataset_name, self.n_cls) 
            self.is_linear_head = False
            logit_head.weight.data = self.norm_text_features
            self.cls_head = LogitHead(
                            logit_head,
                            logit_scale=4,  
                        ).cuda()
        else:
            linear_head = nn.Linear(512, self.n_cls)  
            self.text_features = get_no_norm_512_linear_head_weight(dataset_name, self.n_cls)
            self.is_linear_head = True
            linear_head.weight.data = self.text_features
            self.cls_head = linear_head


    def forward(self, image, label=None):
        cls_token, x_f, x_512 = self.image_encoder(image.type(self.dtype))
        x_f = x_f.permute(1, 0, 2)  
        trans_x = self.trans_block(x_f)  
        image_features = cls_token + trans_x*self.alpha

        x_meta = trans_x / trans_x.norm(dim=-1, keepdim=True)
        prompts = self.prompt_learner(x_meta)  

        bs = image.size(0)
        bs_tokenized_prompts = self.tokenized_prompts.expand(bs, -1) 
        text_features = self.text_encoder(prompts, bs_tokenized_prompts)

        if self.prompt_learner.training:

            classname = [self.dm.lab2cname[lab.item()] for lab in label]
            classname = [clsname.replace("_", " ") for clsname in classname]  
            temp = CUSTOM_TEMPLATES[self.dataset]
            hard_prompt = [temp.format(cls) for cls in classname]  
            tokenized_hard_prompt = torch.cat([clip.tokenize(p) for p in hard_prompt])
            tokenized_hard_prompt = tokenized_hard_prompt.to('cuda') 
            one_hot_labels = torch.nn.functional.one_hot(tokenized_hard_prompt, num_classes=49408) 

            prompt_t = prompts  
   
            score = prompt_t @ self.embedding_weight_t  
            vocab = torch.softmax(score, dim=-1)  

            prompt_loss = F.cross_entropy(vocab.type(self.dtype), one_hot_labels.type(self.dtype))*100  

            concat_features = torch.cat((image_features, text_features), dim=0) 
            concat_label = torch.cat([label, label], dim=0) 
            logits = self.cls_head(concat_features) 
            cls_loss = F.cross_entropy(logits, concat_label)

            if self.is_linear_head:

                hard_text_features = self.text_features.cuda()
                similarity = F.cosine_similarity(image_features, hard_text_features, dim=-1)
                pos_sim = similarity[label].exp()
                neg_sim = similarity.exp().sum() - pos_sim
                contrast_loss = -torch.log(pos_sim / neg_sim )
                hard_text_feature = hard_text_features[label]
                norm_text_features = text_features / text_features.norm(dim=-1, keepdim=True)  
                norm_hard_text_features = hard_text_feature / hard_text_feature.norm(dim=-1, keepdim=True)
                cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
                score = cos(norm_text_features, norm_hard_text_features)
                kgcoop_loss = 1.0-torch.mean(score) 


            else:

                norm_hard_text_features = self.norm_text_features.cuda()
                norm_image_feature = F.normalize(image_features, dim=-1)
                similarity = F.cosine_similarity(norm_image_feature, norm_hard_text_features, dim=-1)
                pos_sim = similarity[label].exp()
                neg_sim = similarity.exp().sum() - pos_sim
                contrast_loss = -torch.log(pos_sim / neg_sim )
                norm_hard_text_feature = norm_hard_text_features[label]
                norm_text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                cos = torch.nn.CosineSimilarity(dim=1,eps=1e-07)
                score = cos(norm_text_features, norm_hard_text_feature)
                kgcoop_loss = 1.0-torch.mean(score) 

            loss =  prompt_loss + cls_loss + kgcoop_loss*8.0 + contrast_loss 
            return loss, prompt_loss, cls_loss, kgcoop_loss, contrast_loss
        

        logits = self.cls_head(image_features)
        return logits 
    
    


@TRAINER_REGISTRY.register()
class EFF_Prompts(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.EFF_PROMPTS.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.EFF_PROMPTS.PREC == "fp32" or cfg.TRAINER.EFF_PROMPTS.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, self.dm, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        prompt_to_update = "prompt_learner"
        class_to_update = "cls_head"
        trans_to_update = "trans_block"
        embedding_to_update = "embedding_weight_t"
        
        for name, param in self.model.named_parameters():
            if (prompt_to_update not in name) and (class_to_update not in name) and (trans_to_update not in name) and (embedding_to_update not in name):
                param.requires_grad_(False)
        for name, param in self.model.image_encoder.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        for name, param in self.model.text_encoder.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM) 
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("efficient_prompt", self.model, self.optim, self.sched)
        
        self.scaler = GradScaler() if cfg.TRAINER.EFF_PROMPTS.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler
        
        prec = self.cfg.TRAINER.EFF_PROMPTS.PREC
        if prec == "amp":
            with autocast():
                loss, prompt_loss, cls_loss, kgcoop_loss ,contrast_loss= model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss, prompt_loss, cls_loss, kgcoop_loss, contrast_loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {
            "loss": loss.item(),
            "cls_loss": cls_loss.item(),
            "prompt_loss": prompt_loss.item(),  
            "kgcoop_loss": kgcoop_loss.item(),
            "contrast_loss": contrast_loss.item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


    #  search
    def search_alpha(self):
        search_scale = 2   
        search_step = 10
        best_search_acc = 0
        best_test_acc = 0
        before = self.model.alpha 
        alpha_list = [i * (search_scale - 0.1) / search_step + 0.1 for i in range(search_step)]
        for alpha in alpha_list:
            with torch.no_grad():
                self.model.alpha = alpha
                acc = self.test(split="val")
            if acc >= best_search_acc:
                best_search_acc = acc
                best_alpha = alpha
                best_test_acc = self.test("test")
        self.model.alpha = before 
        return best_alpha
    

    def after_train(self):
        print("Finish training")
        
        do_test = not self.cfg.TEST.NO_TEST
 
        self.model.alpha = self.search_alpha()
        self.save_model(self.epoch, self.output_dir)
        if do_test:
            print("Deploy the last-epoch model")
            self.test()

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()


    