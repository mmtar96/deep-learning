import flowersclf as clf
import json
import torch
import torch.nn as nn

# data directories
train_dir = 'data/train'
valid_dir = 'data/valid'
test_dir = 'data/test'

# data names of flowers
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
print(cat_to_name)
flowers_type = len(cat_to_name)

loaders, data = clf.ftdata(train_dir, valid_dir, test_dir)

model = clf.create_model(loaders=loaders, model="vgg16", output_features=flowers_type)
model = clf.passgpu(model)

criterio = nn.CrossEntropyLoss()
lr = 0.002
optimiser = torch.optim.SGD(model.parameters(), lr=lr)
epochs = 5

m1_save_name = 'modelvgg16v1.pt'
model, training_data = clf.fit(model, loaders, epochs, optimiser, criterio, m1_save_name, checkpoint=False)
acc = clf.accuracy(model, loaders)
clf.plot_results()

img_path = test_dir
clf.predict_flower(img_path, model, cat_to_name, data[0])
clf.rdmtest(img_path, model, data[0], cat_to_name)

# learn by model transference
filename = 'model_vgg16.pt'
model, train_data, o = clf.load_model(filename, model)
