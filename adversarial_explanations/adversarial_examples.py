import torch
from tqdm import tqdm 
from scipy.stats import spearmanr 
from torch.optim.lr_scheduler import MultiplicativeLR

def get_adversarial_example(model, example, explainer, max_steps=500, alpha=1.0, beta=1.0, 
                            max_eps=2.0, k_percent=0.15, threshold=1e-5, patience=20, optimize_pred=True, 
                            num_lr_steps=200, lr=.1, decay_factor=0.5):
    for key in example:
        example[key] = example[key].to(explainer.device)
    inputs_embeds = model.get_input_embeddings()(example["input_ids"]).unsqueeze(0)
    batch_example = {key:val.unsqueeze(0) for key,val in example.items()}
    args = {key:val for key,val in batch_example.items() if key != "input_ids"}
    model_out = model(**args, inputs_embeds=inputs_embeds) 
    pred = torch.argmax(model_out.logits)
    explanation = explainer.get_explanations(batch_example, model_out, inputs_embeds)
    eps = torch.rand_like(inputs_embeds)
    optimizer = torch.optim.Adam([eps],lr=lr)
    scheduler = MultiplicativeLR(optimizer,lambda epoch: decay_factor)
    attributions = explanation["word_attributions"].squeeze(0)
    attributions = torch.abs(attributions) / torch.sum(torch.abs(attributions))
    k = int(len(attributions)*k_percent)
    top_vals, top_indices = torch.topk(attributions, k)
    attribution_top_sum = torch.sum(attributions[top_indices])

    eps.requires_grad = True 
    eps.retain_grad()
    if not optimize_pred:
        adversarial_examples = [] 
        adversarial_logits = [] 
        adversarial_explanations = [] 
        adversarial_scores = [] 
    early_stop_counter = 0 
    for i in range(max_steps):
        optimizer.zero_grad() 
        eps.grad = None 
        model.zero_grad() 
        old_eps = eps.detach().clone()
        clamp_eps = torch.clamp(eps, min=-max_eps, max=max_eps)
        inputs_embeds = model.get_input_embeddings()(example["input_ids"]).unsqueeze(0)
        model_out = model(**args, inputs_embeds=inputs_embeds) 
        pred = torch.argmax(model_out.logits)
        explanation = explainer.get_explanations(batch_example, model_out, inputs_embeds)
        attributions = explanation["word_attributions"].squeeze(0)
        attributions = torch.abs(attributions) / torch.sum(torch.abs(attributions))

        perturb_embeds = inputs_embeds + clamp_eps 
        perturb_out = model(**args, inputs_embeds=perturb_embeds)
        perturb_explanation = explainer.get_explanations(batch_example, perturb_out, perturb_embeds)
        perturb_attributions = perturb_explanation["word_attributions"].squeeze(0) 
        perturb_attributions = torch.abs(perturb_attributions)/torch.sum(torch.abs(perturb_attributions))
        perturb_top_sum = torch.sum(perturb_attributions[top_indices]) / top_indices.shape[0]
        if not optimize_pred:
            adversarial_examples.append(perturb_embeds.detach().cpu())
            adversarial_logits.append(perturb_out.logits.detach().cpu())
            adversarial_explanations.append(perturb_attributions)
            adversarial_scores.append(perturb_top_sum.detach().cpu())
        if perturb_top_sum.requires_grad != True:
            perturb_top_sum.requires_grad = True 
        if perturb_attributions.requires_grad != True:
            perturb_attributions.requires_grad = True 
        if optimize_pred:
            loss_val = alpha*torch.square((model_out.logits[:,pred]-perturb_out.logits[:,pred]).squeeze()) + beta*(perturb_top_sum)
        else:
            loss_val = perturb_top_sum
        loss_val.backward() 
        optimizer.step()
        if torch.mean(torch.abs(old_eps - eps)) < threshold:
            early_stop_counter += 1 
        else:
            early_stop_counter = 0 
        if early_stop_counter >= patience:
            break 

        if i != 0 and i % num_lr_steps == 0:
            scheduler.step()

    if not optimize_pred:
        perturb_preds = torch.tensor([torch.argmax(adversarial_logits[i]) for i in range(len(adversarial_logits))])
        sort_score_inds = torch.argsort(torch.tensor(adversarial_scores))
        top_example_ind = None 
        for i in range(sort_score_inds.shape[0]):
            if perturb_preds[sort_score_inds[i]].item() != pred.item():
                continue 
            else:
                top_example_ind = sort_score_inds[i]
                break
        # If the optimization process fails to find any samples that match the prediction of the original
        # then we take the values of the original as a (very bad) adversarial explanation. 
        if top_example_ind == None:
            adv_logits = model_out.logits
            adv_attributions = attributions
        else:
            adv_logits = adversarial_logits[top_example_ind]
            adv_attributions = adversarial_explanations[top_example_ind]
        result = {} 
        result["eps"] = eps.detach().cpu()
        result["adv_attributions"] = adv_attributions.detach().cpu() 
        result["attributions"] = attributions.detach().cpu()
        result["example"] = example 
        result["logits"] = model_out.logits.detach().cpu()
        result["adv_logits"] = adv_logits.detach().cpu()
    else:
        result = {} 
        result["eps"] = eps.detach().cpu() 
        result["adv_attributions"] = perturb_attributions.detach().cpu()
        result["attributions"] = attributions 
        result["example"] = example 
        result["logits"] = model_out.logits.detach().cpu()
        result["adv_logits"] = perturb_out.logits.detach().cpu()
    return result 
