from dotmap import DotMap
from mne.decoding import CSP
from utils import *
from function import *
from model_module import *
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
np.set_printoptions(suppress=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class DBPNet(nn.Module):
    def __init__(self, in_channels):
        super(DBPNet, self).__init__()
        self.fre = FRBNet()
        self.tem = TABNet(in_channels)
        self.linear = nn.Linear(8, 2)
    def forward(self, x1, x2):
        seq = self.tem(x1)
        fre = self.fre(x2)
        x = torch.cat((seq, fre), dim=1)
        x = self.linear(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class CustomDatasets(Dataset):
    # initialization: data and label
    def __init__(self, seq_data, fre_data, event_data):
        self.seq_data = seq_data
        self.fre_data = fre_data
        self.label = event_data

    # get the size of data
    def __len__(self):
        return len(self.label)

    # get the data and label
    def __getitem__(self, index):
        seq_data = torch.Tensor(self.seq_data[index])
        fre_data = torch.Tensor(self.fre_data[index])
        # label = torch.LongTensor(self.label[index])
        label = torch.Tensor(self.label[index])

        return seq_data, fre_data, label

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 训练前初始化配置
def initiate(args, train_loader, valid_loader, test_loader, subject):
    model = DBPNet(args.csp_comp)

    # 打印模型参数量
    print(model)
    print(f"The model has {count_parameters(model):,} trainable parameters.")

    # 获取损失函数
    criterion = nn.CrossEntropyLoss()

    # 获取优化器
    optimizer = optim.AdamW(params=model.parameters(), lr=0.003, weight_decay=3e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.003 / 10)
    model = model.cuda()
    criterion = criterion.cuda()

    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}

    return train_model(settings, args, train_loader, valid_loader, test_loader, subject)

def train_model(settings, args, train_loader, valid_loader, test_loader, subject):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']

    def train(model, optimizer, criterion, scheduler):
        model.train()
        proc_loss, proc_size = 0, 0
        num_batches = int(args.n_train // args.batch_size)
        train_acc_sum = 0
        train_loss_sum = 0
        for i_batch, batch_data in enumerate(train_loader):
            seq_data, fre_data, train_label = batch_data
            train_label = train_label.squeeze(-1)
            seq_data, fre_data, train_label = seq_data.cuda(), fre_data.cuda(), train_label.cuda()

            batch_size = train_label.size(0)

            # Forward pass
            preds = model(seq_data, fre_data)
            loss = criterion(preds, train_label.long())
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            proc_loss += loss.item() * batch_size
            proc_size += batch_size
            train_loss_sum += loss.item() * batch_size
            predicted = preds.data.max(1)[1]
            train_acc_sum += predicted.eq(train_label).cpu().sum()

            if i_batch % args.log_interval == 0 and i_batch > 0 and i_batch < num_batches:
                avg_loss = proc_loss / proc_size
                avg_acc = train_acc_sum / proc_size
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Train Loss {:5.4f} | Train Acc {:5.4f}'.
                      format(epoch, i_batch, num_batches, avg_loss, avg_acc))
                proc_loss, proc_size, train_acc_sum = 0, 0, 0

        scheduler.step()

        return train_loss_sum / args.n_train

    def evaluate(model, criterion, test=False):
        model.eval()
        if test:
            loader = test_loader
            num_batches = int(args.n_test // args.batch_size)
        else:
            loader = valid_loader
            num_batches = int(args.n_valid // args.batch_size)
        total_loss = 0.0
        test_acc_sum = 0
        proc_size = 0

        with torch.no_grad():
            for i_batch, batch_data in enumerate(loader):
                seq_data, fre_data, test_label = batch_data
                test_label = test_label.squeeze(-1)
                seq_data, fre_data, test_label = seq_data.cuda(), fre_data.cuda(), test_label.cuda()

                proc_size += args.batch_size
                preds = model(seq_data, fre_data)
                # Backward and optimize
                optimizer.zero_grad()

                total_loss += criterion(preds, test_label.long()).item() * args.batch_size

                predicted = preds.data.max(1)[1]  # 32
                test_acc_sum += predicted.eq(test_label).cpu().sum()

        avg_loss = total_loss / (num_batches * args.batch_size)

        avg_acc = test_acc_sum / (num_batches * args.batch_size)

        return avg_loss, avg_acc

    best_epoch = 1
    best_valid = float('inf')
    for epoch in range(1, args.max_epoch + 1):
        train_loss = train(model, optimizer, criterion, scheduler)
        val_loss, val_acc = evaluate(model, criterion, test=False)

        print("-" * 50)
        print(
            'Epoch {:2d} Finsh | Subject {} | Train Loss {:5.4f} | Valid Loss {:5.4f} | Valid Acc {:5.4f}'.format(epoch,
                                                                                                                  args.subject_number,
                                                                                                                  train_loss,
                                                                                                                  val_loss,
                                                                                                                  val_acc))
        print("-" * 50)
        if val_loss < best_valid:
            best_valid = val_loss

            best_epoch = epoch
            print(f"Saved model at pre_trained_models/{save_load_name(args, name=args.name)}.pt!")
            save_model(args, model, name=args.name)
            stale = 0
        else:
            stale += 1
            if stale > args.patience:
                print(f"Early stopping at epoch {epoch}!")
                break

    model = load_model(args, name=args.name)
    test_loss, test_acc = evaluate(model, criterion, test=True)
    print(f'Best epoch: {best_epoch}')
    print(f"Subject: {subject}, Acc: {test_acc:.2f}")

    return test_loss, test_acc

def main(name="S16", data_document_path="./asset/", length = 5):
    args = DotMap()
    args.name = name
    args.subject_number = int(args.name[1:])
    args.data_document_path = data_document_path
    args.ConType = ["No"]
    args.fs = 128
    args.window_length = math.ceil(args.fs * length)
    args.overlap = 0.5
    args.batch_size = 32
    args.max_epoch = 200
    args.patience = 15
    # args.random_seed = time.time()
    args.log_interval = 20
    args.image_size = 32
    args.people_number = 16
    args.eeg_channel = 64
    args.audio_channel = 1
    args.channel_number = args.eeg_channel + args.audio_channel * 2
    args.trail_number = 8
    args.cell_number = 46080
    args.test_percent = 0.1
    args.vali_percent = 0.1
    args.csp_comp = 32           #select appropriate value
    args.label_col = 0

    args.delta_low = 1
    args.delta_high = 3
    args.theta_low = 4
    args.theta_high = 7
    args.alpha_low = 8
    args.alpha_high = 13
    args.beta_low = 14
    args.beta_high = 30
    args.gamma_low = 31
    args.gamma_high = 50
    args.log_path = "./result"


    args.frequency_resolution = args.fs / args.window_length
    args.point0_low = math.ceil(args.delta_low / args.frequency_resolution)
    args.point0_high = math.ceil(args.delta_high / args.frequency_resolution) + 1
    args.point1_low = math.ceil(args.theta_low / args.frequency_resolution)
    args.point1_high = math.ceil(args.theta_high / args.frequency_resolution) + 1
    args.point2_low = math.ceil(args.alpha_low / args.frequency_resolution)
    args.point2_high = math.ceil(args.alpha_high / args.frequency_resolution) + 1
    args.point3_low = math.ceil(args.beta_low / args.frequency_resolution)
    args.point3_high = math.ceil(args.beta_high / args.frequency_resolution) + 1
    args.point4_low = math.ceil(args.gamma_low / args.frequency_resolution)
    args.point4_high = math.ceil(args.gamma_high / args.frequency_resolution) + 1
    args.window_metadata = DotMap(start=0, end=1, target=2, index=3, trail_number=4, subject_number=5)
    logger = get_logger(args.name, args.log_path, length)

    # load data 和 label
    eeg_data, event_data = read_prepared_data(args)
    data = np.vstack(eeg_data)
    eeg_data = data.reshape([args.trail_number, -1, args.eeg_channel])
    event_data = np.vstack(event_data)

    train_eeg, test_eeg, train_label, test_label = sliding_window(eeg_data, event_data, args, args.eeg_channel)

    # fft
    train_data0 = to_alpha0(train_eeg, args)
    test_data0 = to_alpha0(test_eeg, args)
    train_data1 = to_alpha1(train_eeg, args)
    test_data1 = to_alpha1(test_eeg, args)
    train_data2 = to_alpha2(train_eeg, args)
    test_data2 = to_alpha2(test_eeg, args)
    train_data3 = to_alpha3(train_eeg, args)
    test_data3 = to_alpha3(test_eeg, args)
    train_data4 = to_alpha4(train_eeg, args)
    test_data4 = to_alpha4(test_eeg, args)

    # tf.split()

    train_data0 = gen_images(train_data0, args)
    test_data0 = gen_images(test_data0, args)
    train_data1 = gen_images(train_data1, args)
    test_data1 = gen_images(test_data1, args)
    train_data2 = gen_images(train_data2, args)
    test_data2 = gen_images(test_data2, args)
    train_data3 = gen_images(train_data3, args)
    test_data3 = gen_images(test_data3, args)
    train_data4 = gen_images(train_data4, args)
    test_data4 = gen_images(test_data4, args)

    input_train_data = np.stack([train_data0, train_data1, train_data2, train_data3, train_data4], axis=1)
    test_data = np.stack([test_data0, test_data1, test_data2, test_data3, test_data4], axis=1)

    fre_train_data = np.expand_dims(input_train_data, axis=-1)
    fre_test_data = np.expand_dims(test_data, axis=-1)

    eeg_data = eeg_data.transpose(0, 2, 1)
    eeg_data = eeg_data[:, :args.eeg_channel, :]
    label = np.array(event_data)
    label = np.squeeze(label - 1)
    csp = CSP(n_components=args.csp_comp, reg=None, log=None, cov_est='concat', transform_into='csp_space', norm_trace=True)
    eeg_data = csp.fit_transform(eeg_data, label)
    eeg_data = eeg_data.transpose(0, 2, 1)

    train_eeg, test_eeg, train_label, test_label = sliding_window(eeg_data, label, args, args.csp_comp)
    seq_train_data = np.expand_dims(train_eeg, axis=-1)
    seq_test_data = np.expand_dims(test_eeg, axis=-1)
    del data

    np.random.seed(200)
    np.random.shuffle(fre_train_data)
    np.random.seed(200)
    np.random.shuffle(seq_train_data)
    np.random.seed(200)
    np.random.shuffle(train_label)

    np.random.seed(200)
    np.random.shuffle(fre_test_data)
    np.random.seed(200)
    np.random.shuffle(seq_test_data)
    np.random.seed(200)
    np.random.shuffle(test_label)

    seq_train_data, seq_valid_data, fre_train_data, fre_valid_data, train_label, valid_label = train_test_split(seq_train_data, fre_train_data, train_label, test_size=0.1, random_state=42)

    args.n_train = np.size(train_label)
    args.n_valid = np.size(valid_label)
    args.n_test = np.size(test_label)

    fre_train_data = fre_train_data.transpose(0, 4, 1, 2, 3)
    fre_valid_data = fre_valid_data.transpose(0, 4, 1, 2, 3)
    fre_test_data = fre_test_data.transpose(0, 4, 1, 2, 3)

    seq_train_data = seq_train_data.transpose(0, 3, 2, 1)
    seq_valid_data = seq_valid_data.transpose(0, 3, 2, 1)
    seq_test_data = seq_test_data.transpose(0, 3, 2, 1)


    train_loader = DataLoader(dataset=CustomDatasets(seq_train_data, fre_train_data, train_label),
                                  batch_size=args.batch_size, drop_last=True)
    valid_loader = DataLoader(dataset=CustomDatasets(seq_valid_data, fre_valid_data, valid_label),
                                  batch_size=args.batch_size, drop_last=True)
    test_loader = DataLoader(dataset=CustomDatasets(seq_test_data, fre_test_data, test_label),
                                 batch_size=args.batch_size, drop_last=True)

    # 训练
    loss, acc = initiate(args, train_loader, valid_loader, test_loader, args.name)

    print(loss, acc)
    logger.info(loss)
    logger.info(acc)


if __name__ == "__main__":
    main()
