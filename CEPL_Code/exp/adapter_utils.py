from tqdm import tqdm
import torch
import torch.nn.functional as F
import os

# from Tip-Adapter alpha beta scale(beta,alpha)
lst = {'caltech101':[3, 1, [12, 5]], 'dtd':[2, 1, [13, 13]], \
       'eurosat':[2, 1, [12, 10]], 'fgvc':[5, 1, [30, 30]], \
        'food101':[1.17, 1, [10, 10]], 'imagenet':[1.17, 1, [7, 3]], \
        'oxford_flowers':[10, 1, [50, 50]], 'oxford_pets':[1.17, 1, [7, 3]], \
        'stanford_cars' : [3, 1, [20, 10]], 'sun397' : [1.17, 1, [12, 10]], \
        'ucf101':[3, 1, [7, 3]] }

def build_cache_model(cfg, clip_model, train_loader_cache):
    path = 'cache/' + cfg.DATASET.NAME + '/' + str(cfg.DATASET.NUM_SHOTS)+'shots_seed'+ str(cfg.SEED)
    if os.path.exists(path) is False:
        os.makedirs(path)
        cache_keys = []
        cache_values = []
        clip_model.visual.to('cuda')
        with torch.no_grad():
            for augment_idx in range(10):
                train_features = []
                print('Augment Epoch: {:} / {:}'.format(augment_idx, 10))
                for i, batch in enumerate(tqdm(train_loader_cache)):
                    images = batch['img']
                    images = images.to('cuda')
                    image_features, x_f, x_512  = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = batch['label']
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0))
        
        torch.save(cache_keys, path + '/keys_' + str(cfg.DATASET.NUM_SHOTS) + "shots.pt")
        torch.save(cache_values, path + '/values_' + str(cfg.DATASET.NUM_SHOTS) + "shots.pt")
    else:
        cache_keys = torch.load(path + '/keys_' + str(cfg.DATASET.NUM_SHOTS) + "shots.pt")
        cache_values = torch.load(path + '/values_' + str(cfg.DATASET.NUM_SHOTS) + "shots.pt")
    return cache_keys, cache_values


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

def search(cfg, model, val_loader):
 
    org_cache, new_cache, labels = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            images, target = batch['img'].cuda(), batch['label'].cuda()
            im_feat_org, im_feat_new = model.image_encoder(images.type(model.dtype))
            im_feat_org = im_feat_org / im_feat_org.norm(dim=-1, keepdim=True)
            im_feat_new = im_feat_new / im_feat_new.norm(dim=-1, keepdim=True)
                
            org_cache.append(im_feat_org)
            new_cache.append(im_feat_new)
            labels.append(target)
    org_cache, new_cache, labels = torch.cat(org_cache), torch.cat(new_cache), torch.cat(labels)


       

    search_step = [200, 20]
    search_scale = lst[cfg.DATASET.NAME][2:] if cfg.DATASET.NAME in lst else [7, 3]
    beta_list = [i * (search_scale[0] - 0.1) / search_step[0] + 0.1 for i in range(search_step[0])]
    alpha_list = [i * (search_scale[1] - 0.1) / search_step[1] + 0.1 for i in range(search_step[1])]
    best_acc = 0
    best_beta, best_alpha = 0, 0
       
    for beta in beta_list:
        for alpha in alpha_list:
            with torch.no_grad():
                affinity = model.adapter(new_cache)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ model.cache_values

            # clip_logits
            logit_scale = model.logit_scale.exp()
            clip_logits = logit_scale * org_cache @ model.text_feat.t()

            # sum
            logits = clip_logits + cache_logits * alpha
            acc = cls_acc(logits, labels)
            
            if acc > best_acc:
                print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, acc))
                best_acc = acc
                best_beta = beta
                best_alpha = alpha

    return best_alpha, best_beta




