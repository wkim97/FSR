'''
Codes modified from original codes of:
CIFS : https://github.com/HanshuYAN/CIFS
'''


import torch


def defense_success_rate(predict, loader, attack_class, attack_kwargs,
                         device=torch.device("cuda:0"), num_batch=None):
    
    adversary = attack_class(predict, **attack_kwargs)
    accuracy, defense_success_rate, defense_success, natural_success = \
        attack_mini_batches(predict, adversary, loader, device=device, num_batch=num_batch)

    message = 'Ori Acc: {:.2f}%\tAdv Acc: {:.2f}%'.format(accuracy*100, defense_success_rate*100)

    return message, defense_success, natural_success


def predict_from_logits(logits, dim=1):
    return logits.topk(1, dim)[1]


def attack_mini_batches(predict, adversary, loader, device="cuda", num_batch=None, topk=1):
    lst_label = []
    lst_pred = []
    lst_advpred = []

    idx_batch = 0
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        adv = adversary.perturb(data, label)
        
        adv_logits, _, _, _ = predict(adv, is_eval=True)
        advpred = predict_from_logits(adv_logits)
        nat_logits, _, _, _ = predict(data, is_eval=True)
        pred = predict_from_logits(nat_logits)

        lst_label.append(label)
        lst_pred.append(pred)
        lst_advpred.append(advpred)
            
        idx_batch += 1
        if idx_batch == num_batch:
            break
    
    label = torch.cat(lst_label).view(-1, 1)
    pred = torch.cat(lst_pred).view(-1, topk)
    advpred = torch.cat(lst_advpred).view(-1, topk)
    
    num = label.size(0)
    accuracy = (label == pred).sum().item() / num
    defense_success = (label == advpred)
    natural_success = (label == pred)
    def_success_rate = (label == advpred).sum().item() / num
    
    return accuracy, def_success_rate, defense_success, natural_success
