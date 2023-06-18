
# import libararies

import torch
from torch import nn
from torch.optim import Adam

from tqdm import tqdm

from utils.dataset import Dataset


class TrainEvaluate:

    def __init__(self, model):
         self._model = model

    def train(self, train_data, val_data, learning_rate, epochs):

        train, val = Dataset(train_data), Dataset(val_data)
        model = self._model

        train_dataloader = torch.utils.data.DataLoader(train, batch_size=15, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val, batch_size=15)

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            print("Cleaning cache...")
            torch.cuda.empty_cache()

            print("Using GPU for training...")
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        criterion = nn.CrossEntropyLoss()
        non_frozen_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = Adam(non_frozen_params, lr= learning_rate)

        if use_cuda:

                model = model.cuda()
                criterion = criterion.cuda()

        for epoch_num in range(epochs):

                total_acc_train = 0
                total_loss_train = 0

                for train_input, train_label in tqdm(train_dataloader):

                    train_label = train_label.to(device)
                    mask = train_input['attention_mask'].to(device)
                    input_id = train_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)
                    
                    batch_loss = criterion(output, train_label.long())
                    total_loss_train += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == train_label).sum().item()
                    total_acc_train += acc

                    model.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                
                total_acc_val = 0
                total_loss_val = 0

                with torch.no_grad():

                    for val_input, val_label in val_dataloader:

                        val_label = val_label.to(device)
                        mask = val_input['attention_mask'].to(device)
                        input_id = val_input['input_ids'].squeeze(1).to(device)

                        output = model(input_id, mask)

                        batch_loss = criterion(output, val_label.long())
                        total_loss_val += batch_loss.item()
                        
                        acc = (output.argmax(dim=1) == val_label).sum().item()
                        total_acc_val += acc
                
                print(
                    f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                    | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                    | Val Loss: {total_loss_val / len(val_data): .3f} \
                    | Val Accuracy: {total_acc_val / len(val_data): .3f}')
    
    def evaluate(self, test_data):

        test = Dataset(test_data)

        test_dataloader = torch.utils.data.DataLoader(test, batch_size=10)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        if use_cuda:

            model = self.model.cuda()

        total_acc_test = 0
        with torch.no_grad():

            for test_input, test_label in test_dataloader:

                test_label = test_label.to(device)
                mask = test_input['attention_mask'].to(device)
                input_id = test_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                acc = (output.argmax(dim=1) == test_label).sum().item()
                total_acc_test += acc
        
        print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')
