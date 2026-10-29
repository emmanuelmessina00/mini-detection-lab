
def train_one_epoch(model,optimizer,dataloader,device):
    model.train()
    total_loss = 0

    for images, labels in dataloader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in labels]

        loss_dict = model(images, targets)
        losses= sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss/len(dataloader)