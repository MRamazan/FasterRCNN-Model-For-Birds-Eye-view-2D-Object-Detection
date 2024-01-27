import torchvision
from torchvision.models.detection.faster_rcnn import  FasterRCNN_ResNet50_FPN_Weights
from torchvision import  models
from VisdroneDataset import VisdroneDataset
import torch
from torch.utils import data
import os




def main():

    train_path = r"C:\Users\PC\Downloads\VisDrone2019-DET-train\VisDrone2019-DET-train"
    batch_size = 10
    epochs = 5
    dataset = VisdroneDataset(train_path)

    def collate_fn(batch):
        return tuple(zip(*batch))
    dataloader = data.DataLoader(dataset, shuffle=True,batch_size=10, collate_fn=collate_fn)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    in_features = model.roi_heads.box_predictor.cls_score.in_features  # we need to change the head
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, 11)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.001)


    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    total_num_batches = int(len(dataset)) / batch_size
    print("total data: ", len(dataset))

    model_path = os.path.abspath(os.path.dirname(__file__)) + '/weights/'
    latest_model = None
    first_epoch = 0
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    else:
        try:
            latest_model = [x for x in sorted(os.listdir(model_path)) if x.endswith('.pkl')][-1]
        except:
            pass

    if latest_model is not None:
        checkpoint = torch.load(model_path + latest_model)
        print(checkpoint['optimizer_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        first_epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        print('Found previous checkpoint: %s at epoch %s' % (latest_model, first_epoch))
        print('Resuming training....')

    for epoch in range(first_epoch, epochs+1):
        model.train()
        all_losses = []
        all_losses_dict = []
        curr_batch = 0
        passes = 0
        for count, batch in enumerate(dataloader):
            images, targets = batch

            images = [image.cpu() for image in images]
            targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
            loss_value = losses.item()

            all_losses.append(loss_value)
            all_losses_dict.append(loss_dict_append)



            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            print(count)
            if (passes % 5

                    == 0):
                print("--- epoch %s | batch %s/%s --- [loss: %s]" %(epoch, curr_batch, total_num_batches, losses.item()))
                passes = 0

            passes += 1
            curr_batch += 1

        if epoch % 1 == 0:


             name = model_path + 'epoch_%s.pkl' % epoch
             print("====================")
             print("Done with epoch %s!" % epoch)
             print("Saving weights as %s ..." % name)
             torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses
             }, name)
             print("====================")



if __name__ == '__main__':
    main()


















