


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

            images = images.clone().detach() #o(device)
            labels = labels#to(device)
    #         print(images.shape)
            bs, _, _, _ = images.shape

    #         print(bs)
            optimizer.zero_grad()

            preds = model(images)
    #         print(bs)
            preds_lengths = torch.full(size=(bs,), fill_value=total_len, dtype=torch.long)
            target_lengths = torch.randint(low=1, high=total_len, size=(bs,), dtype=torch.long)
    #         print(torch.max(preds))
    #         break
            loss = criterion(preds, labels, preds_lengths, target_lengths)

    #         print(loss)
    #         break
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
    #         break
            if steps % print_every == 0:
                all_losses.append(running_loss / print_every)
                model.eval() # incase theres dropout, turn dropout off

    #             with torch.no_grad():
    #                 val_loss, accuracy = validation(model, val_loader)

                print("Epoch: {}/{}... ".format(e+1, epochs),
                     "Training Loss: {:.3f}.. ".format(running_loss/print_every))
    #                   "Test Loss: {:.3f}... ".format(val_loss/len(val_loader)),
    #                   "Test Accuracy: {:.3f}".format(accuracy/len(val_loader)))

                running_loss = 0

                model.train() # turn dropout back on


            
if main == 'main':
    train()
            
        