import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageEnhance 
import random
import models_vit
from pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_

# Call Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = models_vit.__dict__['vit_large_patch16'](
    num_classes=3,
    drop_path_rate=0.2,
    global_pool=False,
).to(device)

checkpoint = torch.load('RETFound_cfp_weights.pth', map_location=device)
checkpoint_model = checkpoint['model']
state_dict = model.state_dict()
for k in ['head.weight', 'head.bias']:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]

interpolate_pos_embed(model, checkpoint_model)
msg = model.load_state_dict(checkpoint_model, strict=False)


# Data Loader
def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        value = 128 + factor * (c - 128)
        return max(0, min(255, value))
    return img.point(contrast)

def image_loader(image_path, rotation=0, image_size = 224): # Image, GroundTruths
    
    rotation = random.randint(0, 360)
    contrast = random.randint(0, 25)
     
    
    try:
        im = Image.open(image_path)
        im = im.rotate(rotation)
        im = change_contrast(im, contrast)
        
        if np.mean(im) < 30:
            im3 = ImageEnhance.Brightness(im)
            im = im3.enhance(2)
            
        elif np.mean(im) < 40:
            im3 = ImageEnhance.Brightness(im)
            im = im3.enhance(1.5)
        
        im = im.resize((image_size,image_size))
        im = torchvision.transforms.functional.pil_to_tensor(im).reshape([1,3,image_size,image_size])
        im = im / 255
        
        return im

    except FileNotFoundError:
        return False
    
def batch_loader(image_paths, truths, img_size=224):
    
    train_x = torch.zeros(0, 3, img_size, img_size)
    train_y = torch.zeros(0, 3)
    
    for i in range(0, len(image_paths)):
        x = image_loader(image_paths[i], image_size = img_size)
        y = truths[i]
        
        if x is not False:
            train_x = torch.cat((train_x, x), 0)
            train_y = torch.cat((train_y, y), 0)
            
    return train_x, train_y

Shanghai = "Datasets/Shanghai HR/HRDC Hypertensive Retinopathy Classification Training Labels.csv"
ODIR = "Datasets/ODIR/full_df.csv"
messidor = "Datasets/messidor-2 DR/messidor_data.csv"
        
image_list = []
ground_truth_list = []

df_shanghai = pd.read_csv(Shanghai)
df_odir = pd.read_csv(ODIR)
df_messidor = pd.read_csv(messidor)

# ODIR
n=0
for img in range(0, df_odir.shape[0]): # ODIR - LEFT
    
    if df_odir['Left-Diagnostic Keywords'][n] == 'normal fundus':
        image_list.append("Datasets/ODIR/preprocessed_images/" + df_odir['Left-Fundus'][img])
        ground_truth_list.append(torch.tensor([[1.0, 0.0, 0.0]]))
        
    elif sum([df_odir["N"][n], df_odir["D"][n], df_odir["H"][n]]) == 1:
        if 'diabetic' in df_odir['Left-Diagnostic Keywords'][n]:
            image_list.append("Datasets/ODIR/preprocessed_images/" + df_odir['Right-Fundus'][img])
            ground_truth_list.append(torch.tensor([[0.0, 1.0, 0.0]]))
            
        elif 'hyper' in df_odir['Left-Diagnostic Keywords'][n]:
            image_list.append("Datasets/ODIR/preprocessed_images/" + df_odir['Right-Fundus'][img])
            ground_truth_list.append(torch.tensor([[0.0, 0.0, 1.0]]))
 
    n+=1
    
n=0
for img in range(0, df_odir.shape[0]): # ODIR - RIGHT
    
    if df_odir['Right-Diagnostic Keywords'][n] == 'normal fundus':
        image_list.append("Datasets/ODIR/preprocessed_images/" + df_odir['Right-Fundus'][img])
        ground_truth_list.append(torch.tensor([[1.0, 0.0, 0.0]]))
        
    elif sum([df_odir["N"][n], df_odir["D"][n], df_odir["H"][n]]) == 1:
        if 'diabetic' in df_odir['Right-Diagnostic Keywords'][n]:
            image_list.append("Datasets/ODIR/preprocessed_images/" + df_odir['Right-Fundus'][img])
            ground_truth_list.append(torch.tensor([[0.0, 1.0, 0.0]]))
            
        elif 'hyper' in df_odir['Right-Diagnostic Keywords'][n]:
            image_list.append("Datasets/ODIR/preprocessed_images/" + df_odir['Right-Fundus'][img])
            ground_truth_list.append(torch.tensor([[0.0, 0.0, 1.0]]))
 
    n+=1

# SHANGHAI
n=0
for img in df_shanghai['Image']:
    image_list.append("Datasets/Shanghai HR/1-Images/1-Training Set/" + img)
    if df_shanghai["Hypertensive Retinopathy"][n] == 1:
        ground_truth_list.append(torch.tensor([[0.0, 0.0, 1.0]]))
    else:
        ground_truth_list.append(torch.tensor([[1.0, 0.0, 0.0]]))
    n+=1
    
# MESSIDOR
n=0
for img in df_messidor['id_code']:
    if df_messidor["diagnosis"][n] == 0:
        image_list.append("Datasets/messidor-2 DR/preprocess/" + img)
        ground_truth_list.append(torch.tensor([[1.0, 0.0, 0.0]]))
        
    elif df_messidor["diagnosis"][n] > 1:
        image_list.append("Datasets/messidor-2 DR/preprocess/" + img)
        ground_truth_list.append(torch.tensor([[0.0, 1.0, 0.0]]))
    n+=1
    
N = []
D = []
H = []

N_tr = []
D_tr = []
H_tr = []

for i in range(len(image_list)):
    a, b = batch_loader(image_list[i:i+1], ground_truth_list[i:i+1])
    if a.shape != torch.Size([0, 3, 256, 256]):
        if float(nn.MSELoss()(b.float(), torch.tensor([[0., 0., 1.]]))) < 0.1:
            H.append(image_list[i])
            H_tr.append(ground_truth_list[i])
        
        elif float(nn.MSELoss()(b.float(), torch.tensor([[0., 1., 0.]]))) < 0.1:
            D.append(image_list[i])
            D_tr.append(ground_truth_list[i])

        elif float(nn.MSELoss()(b.float(), torch.tensor([[1., 0., 0.]]))) < 0.1:
            N.append(image_list[i])
            N_tr.append(ground_truth_list[i])

n_samples = 5376 # Normal
d_samples = 384 # Diabetic
h_samples = 384 # Hypertension

random.seed(42)
random.shuffle(N)
random.shuffle(D)
random.shuffle(H)

image_list = N[:n_samples] + D[:d_samples]*14 + H[:h_samples]*14
ground_truth_list = N_tr[:n_samples] + D_tr[:d_samples]*14 + H_tr[:h_samples]*14

test_image_list = N[n_samples:n_samples+1344] + D[d_samples:d_samples+96]*14 + H[h_samples:h_samples+96]*14
test_ground_truth_list = N_tr[n_samples:n_samples+1344] + D_tr[d_samples:d_samples+96]*14 + H_tr[h_samples:h_samples+96]*14


# Loss, Optimizer
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.000004)


# Training Loop
model.train()
epochs = 30
batch = 8
data_list = []
j = 0

for i in range(0, len(image_list)):
    data_list.append(i)


# TRAIN & TEST LOOP
for epoch in range (epochs):
    epoch_loss = 0
    count = 0
    random.shuffle(data_list)
    torch.save(model.state_dict(), str(epoch) + "_epochs")
    
    
    # TRAIN
    pred_true = 0
    N_N, N_D, N_H, D_N, D_D, D_H, H_N, H_D, H_H = [0]*9
    model.train()
    
    for i in range(0, len(image_list), batch):
        
        # TRAIN
        x = []
        y = []
        
        for k in range(0, batch):
            x.append(image_list[data_list[j % len(image_list)]])
            y.append(ground_truth_list[data_list[j % len(image_list)]])
            j += 1

        train_x, train_y = batch_loader(x, y)
        train_x = train_x.to(device)
        train_y = train_y.to(device).float()
        
        if train_x.shape != torch.Size([0, 3, 256, 256]):
            
            count += batch
            optimizer.zero_grad()
            predict_y = model(train_x)
            loss = loss_fn(predict_y, train_y)
            loss.backward()
            epoch_loss += float(loss)
            optimizer.step()
            
            for b in range(0, batch):
                if (torch.argmax(train_y[b]).item() == torch.argmax(predict_y[b])).item():
                    pred_true += 1
                    if torch.argmax(predict_y[b]).item() == 0:
                        N_N += 1
                    elif torch.argmax(predict_y[b]).item() == 1:
                        D_D += 1
                    else:
                        H_H += 1

                else:
                    if torch.argmax(train_y[b]).item() == 0:
                        if torch.argmax(predict_y[b]).item() == 1:
                            N_D += 1
                        else:
                            N_H += 1

                    elif torch.argmax(train_y[b]).item() == 1:
                        if torch.argmax(predict_y[b]).item() == 0:
                            D_N += 1
                        else:
                            D_H += 1

                    else:
                        if torch.argmax(predict_y[b]).item() == 0:
                            H_N += 1
                        else:
                            H_D += 1
                        
            if count % 4000 == 0:
                print("-", loss.item())
            #print(N_N, N_D, N_H)
            #print(D_N, D_D, D_H)
            #print(H_N, H_D, H_H)


    # Train Results
    print(epoch+1, ")", round(epoch_loss / (count/batch), 4))
    print((pred_true / count))
    print(N_N, N_D, N_H)
    print(D_N, D_D, D_H)
    print(H_N, H_D, H_H)
            
    
    # EVALUATION
    pred_true = 0
    N_N, N_D, N_H, D_N, D_D, D_H, H_N, H_D, H_H = [0]*9
    model.eval()
    
    for i in range(0, len(test_image_list)):
        train_x, train_y = batch_loader(test_image_list[i:i+1], test_ground_truth_list[i:i+1])
        train_x = train_x.to(device)
        train_y = train_y.to(device).float()
        predict_y = model(train_x)
        
        if train_x.shape != torch.Size([0, 3, 256, 256]):
        
            if (torch.argmax(train_y).item() == torch.argmax(predict_y)).item():
                pred_true += 1
                if torch.argmax(predict_y).item() == 0:
                    N_N += 1
                elif torch.argmax(predict_y).item() == 1:
                    D_D += 1
                else:
                    H_H += 1

            else:
                if torch.argmax(train_y).item() == 0:
                    if torch.argmax(predict_y).item() == 1:
                        N_D += 1
                    else:
                        N_H += 1
                        
                elif torch.argmax(train_y).item() == 1:
                    if torch.argmax(predict_y).item() == 0:
                        D_N += 1
                    else:
                        D_H += 1
                        
                else:
                    if torch.argmax(predict_y).item() == 0:
                        H_N += 1
                    else:
                        H_D += 1
    
    # Test Results
    print((pred_true / len(test_image_list)))
    print(N_N, N_D, N_H)
    print(D_N, D_D, D_H)
    print(H_N, H_D, H_H)
    print("")