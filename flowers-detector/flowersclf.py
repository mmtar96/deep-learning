import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import transforms, models, datasets
import numpy as np
from PIL import Image
import cv2
import random
import matplotlib.pyplot as plt
import time
import copy
from glob import glob

def checkgpu():
    TRAIN_ON_GPU = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if TRAIN_ON_GPU:
        print('CUDA is available. Train on GPU.')
    else:
        print('CUDA unavailable. Train on CPU.')  
    return DEVICE

# fit transform data
def ftdata(train_dir, valid_dir, test_dir):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.2),transforms.RandomPerspective(distortion_scale=0.3, p=0.2),transforms.RandomRotation(10),transforms.Resize((264, 264)),
                                          transforms.CenterCrop((224, 224)),transforms.ToTensor(),normalizer])
    valid_transform = transforms.Compose([transforms.Resize((264, 264)),transforms.CenterCrop((224, 224)),transforms.ToTensor(),normalizer])
    test_transform = transforms.Compose([transforms.Resize((264, 264)),transforms.CenterCrop((224, 224)),transforms.ToTensor(),normalizer])
    
    # data import
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transform)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)
    
    # define dataloaders
    num_workers = 2
    batch = 102
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch,
                                               shuffle=True,
                                               num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=batch,
                                               shuffle=True,
                                               num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch, 
                                              shuffle=True,
                                              num_workers=num_workers)
    loaders = {'train': train_loader,
               'valid': valid_loader,
               'test': test_loader}
    return loaders, (train_data, valid_data, test_data)

def create_model(loaders, model="vgg16", output_features=102):
    device = checkgpu()
    if model == "effnet":
        model_1 = models.efficientnet_b7(pretrained=True)
    else:
        model_1 = models.vgg16(pretrained=True)
    # properties
    architecture = model_1._get_name().lower()
    model_1.architecture = architecture
    model_1.device = device
    print('Model classifier: \n', model_1.classifier)
    input_features_class = model_1.classifier[0].in_features
    print('Number of input features in the classifier: {}'.format(input_features_class))
    print('Number of output features in the classifier: {}'.format(output_features))
    for param in model_1.features.parameters():
        param.requires_grad = False # DESACTIVA EL RECALCULO DE LOS PESOS DE LA RED
    
    custom_classifier_1 = nn.Sequential(nn.Linear(input_features_class, 512, bias=True),
                               nn.ReLU(inplace=True),
                               nn.Dropout(0.3, inplace=False),
                               nn.Linear(512, output_features, bias=True))
    model_1.classifier = custom_classifier_1
    print('Modify clasificator: \n', model_1.classifier)
    return model_1

def passgpu(model): # pass model to GPU
    train_gpu = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device=DEVICE)
    if train_gpu:
        print("Model passed to GPU.")
    else:
        print("Model remains on CPU.")
    return model

def save_model(model, optimiser, train_loss_hist, valid_loss_hist, filename):
    custom_dict = {'model' : model.state_dict(),
                    'optimiser' : optimiser.state_dict(),
                    'train_loss' : train_loss_hist,
                    'valid_loss' : valid_loss_hist,
                    'arquitecture' : model.architecture,
                    'class_to_idx': model.class_to_idx,
                    'device' : model.device}
    torch.save(custom_dict, filename)

def load_model(filename, model):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    model.device = checkpoint['device']
    model.architecture = checkpoint['arquitecture']
    train_loss_hist = checkpoint['train_loss']
    valid_loss_hist = checkpoint['valid_loss']
    for param in model.features.parameters(): 
      param.requires_grad = False
    return model, train_loss_hist, valid_loss_hist

def accuracy(model, loader):
    device = checkgpu()
    num_correct = 0
    num_total = 0
    model.eval()
    with torch.no_grad():
      for i, (x, y) in enumerate(loader):
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.long)
        scores = model(x)
        _, pred = scores.max(dim=1)
        num_correct += (pred == y).sum()
        num_total += pred.size(0)
        result = num_correct / num_total
      return result

def fit(model, loaders, epochs, optimiser, criterio, filename, checkpoint):
    device = checkgpu()
    record_list = []
    valid_loss_min = np.Inf
    train_loss_hist = []
    valid_loss_hist = []
    acc_hist = []

    if checkpoint:
      load_model(filename, model)

    for epoch in range(epochs):
          
        start = time.time()
        
        train_loss = 0
        model.train()
        for i, (x, y) in enumerate(loaders['train']):
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad() 
            output = model(x) 
            loss = criterio(output, y)
            loss.backward() # gradients
            optimiser.step() # backpropagation
            train_loss += loss.item()
        train_loss_hist.append(train_loss)
      
        # validation
        valid_loss = 0
        model.eval()
        for i, (x, y) in enumerate(loaders['valid']):
            x, y = x.to(device), y.to(device)
            with torch.no_grad(): 
                output = model(x).to(device) 
                loss = criterio(output, y)
            valid_loss += loss.item()
        valid_loss_hist.append(valid_loss)

        acc = accuracy(model, loaders['test'])
        acc_hist.append(acc)

        end_time = time.time()
      
        if valid_loss < valid_loss_min:
            save_model(model, optimiser, train_loss_hist, valid_loss_hist, filename)
            valid_loss_min = valid_loss
            best_state_dict = copy.deepcopy(model.state_dict()) # Salvamos el mejor modelo
            epoch_s = epoch
            message = 'Epoch: {}  Training Loss: {:.6f}  Validation Loss: {:.6f}   Time: {:.1f}  Acc_test: {:.6f}  -  SAVED'.format(
                    epoch, train_loss, valid_loss, end_time - start, acc)
            record_list.append(message)    
        else:
            message = 'Epoch: {}  Training Loss: {:.6f}  Validation Loss: {:.6f}   Time: {:.1f} Acc_test: {:.6f}'.format(
                    epoch, train_loss, valid_loss, end_time - start, acc)
            record_list.append(message)  
      
        print(record_list[-1])
        
    model.load_state_dict(best_state_dict)
    print('Checkpoint saved at epoch {}'.format(epoch_s))

    return model, (train_loss_hist, valid_loss_hist, record_list, time.localtime())

def results(training_data):
    tr_losses, vl_losses = training_data[0], training_data[1]
    x = np.arange(1, len(tr_losses))
    fig = plt.figure()
    ax = plt.subplot()
    ax.plot(x, tr_losses, label='training loss')
    ax.plot(x, vl_losses, label='validation loss')
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.set_title("Losses during training")
    plt.legend()
    plt.show()


def predict_flower(img_path, model, cat_to_name, train_data):
    device = checkgpu()
    index_to_name = train_data.class_to_idx
    def get_key(val):
        for key, value in index_to_name.items():
             if val == value:
                 return key
    image = Image.open(img_path)

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((264, 264)),transforms.CenterCrop((224, 224)),transforms.ToTensor(),normalizer])

    image = transform(image)
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor).to(device)

    #print(image.shape)

    with torch.no_grad():
            output = model(image_tensor)

    index_flor = int(torch.argmax(output).cpu().numpy())
    logSoftMax = nn.LogSoftmax(dim=1)
    prob_list = torch.exp(logSoftMax(output))
    #print('Comprobamos que es distribucion de probabilidad: ', float(prob_list.sum()/output.shape[0]))
    prob = torch.max(prob_list)
    print('Probabilidad que ha tenido de acertar de: ', float(prob))
    index_flor = int(torch.argmax(output).cpu().numpy())
    #print('Indice de la neurona que se activa: ', index_flor+1)
    flower_name = cat_to_name[get_key(index_flor)]
    print('El resultado es una flor tipo ... ', flower_name)

    #PLOTEO DE LA IMAGEN
    img = cv2.imread(img_path)
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_image)
    plt.show()

    # End_______________________________________________________________________
    
    return flower_name, prob

def rdmtest(image_path, model, train_data, cat_to_name):
    image_path_arr = random.choices(np.array(glob(image_path)), k=10)
    for image in image_path_arr:
        flower, prob = predict_flower(image, model, train_data.class_to_idx, cat_to_name)
        


