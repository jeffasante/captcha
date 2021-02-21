from torch.utils.data import Dataset, DataLoader



# define transforms
transform = transforms.Compose([transforms.Resize((75, 300)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.73199,), (0.28809,)),
                                ])


# build partion -- train test split
n_data = len(data)
train_size = int(0.9 * n_data)
test_size = n_data - train_size 

full_dataset = CaptchaDataset(data, encoded_targets, transform=transform)

train_dataset, val_dataset = torch.utils.data.random_split(full_dataset,
                                                           [train_size, test_size])

batch_size = 16

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def weights_init(m):
    classname = m.__class__.__name__
    
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)



n_outputs = len(unique_letters)
total_len = 5 # input sequence length

HIDDEN_SIZE = 128

model = CRNN(75, 300, n_outputs, HIDDEN_SIZE) #.to(device)

model.apply(weights_init)

criterion = nn.CTCLoss(zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=0.003)

# # how is our networ doing on new data?
def validation(model, val_loader):
    val_loss = 0
    accuracy = 0
    
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        bs, _, _, _ = images.shape

        optimizer.zero_grad()

        val_preds = model(images)

        val_preds_lengths = torch.full(size=(bs,), fill_value=total_len, dtype=torch.long)
        val_target_lengths = torch.randint(low=1, high=total_len, size=(bs,), dtype=torch.long)
            
        val_loss = criterion(val_preds, labels, val_preds_lengths, val_target_lengths)
        
        val_loss += val_loss.item()
  
        # accuracy
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1]) # to see the index of the highest prob
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return val_loss, accuracy



def train_batch():
    
    epochs = 10
    steps =  0
    print_every = 40
    running_loss = 0

    all_losses = []


    for e in range(epochs):
        model.train()

        for images, labels in train_loader:
            steps += 1

            images = images.clone().detach().to(device)
            labels = labels.to(device)
            
            bs, _, _, _ = images.shape
            
            optimizer.zero_grad()

            preds = model(images)
            
            preds_lengths = torch.full(size=(bs,), fill_value=total_len, dtype=torch.long)
            target_lengths = torch.randint(low=1, high=total_len, size=(bs,), dtype=torch.long)
            
            loss = criterion(preds, labels, preds_lengths, target_lengths)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
    #         break
            if steps % print_every == 0:
                all_losses.append(running_loss / print_every)
                model.eval() # incase theres dropout, turn dropout off

                with torch.no_grad():
                    val_loss, accuracy = validation(model, val_loader)

                print("Epoch: {}/{}... ".format(e+1, epochs),
                     "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}... ".format(val_loss/len(val_loader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(val_loader)))

                running_loss = 0

                model.train() # turn dropout back on


            
if main == 'main':
    train()
            
        