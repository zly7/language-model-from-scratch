

def compute_metrics_for_masklm(logits, labels):
    active_loss_position = (labels != -100)  # There use muktiplier also work
    preds = logits.argmax(-1)
    acc = (preds[active_loss_position] == labels[active_loss_position]).float().mean() if labels is not None else None
    return {"accuracy": acc}