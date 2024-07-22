import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torchquantum import C_DTYPE, switch_little_big_endian_state
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.plugin import (tq2qiskit_expand_params,
                                 tq2qiskit,
                                 tq2qiskit_measurement,
                                 qiskit_assemble_circs)
from torchquantum.plugin import (
    tq2qiskit_measurement,
    qiskit_assemble_circs,
    op_history2qiskit,
    op_history2qiskit_expand_params,
)
from torchquantum.dataset import MNIST
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from ST_Encoder import SpatialDataEncoder


class QFCModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 8
            # gates with trainable parameters
            for i in range(0, 30):
                setattr(self, 'rx' + str(i), tq.RX(has_params=True, trainable=True))
            for i in range(0, 30):
                setattr(self, 'ry' + str(i), tq.RY(has_params=True, trainable=True))
            for i in range(0, 30):
                setattr(self, 'cnot' + str(i), tq.CNOT(has_params=True, trainable=True))

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            """
            1. To convert tq QuantumModule to qiskit or run in the static
            model, need to:
                (1) add @tq.static_support before the forward
                (2) make sure to add
                    static=self.static_mode and
                    parent_graph=self.graph
                    to all the tqf functions, such as tqf.hadamard below
            """
            self.q_device = q_device
            qubits = int(self.n_wires)
            rx_index = 0
            ry_index = 0
            cnot_index = 0
            #This is the First Layer of the Quantum Circuit
            for q in range(0, qubits, 2):
                getattr(self, 'cnot' + str(cnot_index))(self.q_device, wires=[q, q +1])
                cnot_index += 1
            for q in range(0, qubits, 2):
                getattr(self, 'rx' + str(rx_index))(self.q_device, wires=q)
                getattr(self, 'rx' + str(rx_index))(self.q_device, wires=q)
                rx_index += 1
                getattr(self, 'ry' + str(ry_index))(self.q_device, wires=q)
                ry_index += 1
            #This ends the first layer of the QLayer
            #print(qubits)
            #Start of second Layer of the Quantum Circuit
            for q in range(0, (qubits//2)-1):
                getattr(self, 'cnot' + str(cnot_index))(self.q_device, wires=[q, q + 1])
                getattr(self, 'rx' + str(rx_index))(self.q_device, wires=q)
                getattr(self, 'rx' + str(rx_index))(self.q_device, wires=q)
                getattr(self, 'ry' + str(ry_index))(self.q_device, wires=q)
                ry_index += 1
                rx_index += 1
                cnot_index += 1
            # for q in range(0,(qubits//2)-2):
            #     getattr(self, 'rx' + str(rx_index))(self.q_device, wires=q)
            #     rx_index += 1

            for q in range((qubits//2), qubits-1):
                getattr(self, 'cnot' + str(cnot_index))(self.q_device, wires=[q, q + 1])
                getattr(self, 'rx' + str(rx_index))(self.q_device, wires=q)
                getattr(self, 'rx' + str(rx_index))(self.q_device, wires=q)
                getattr(self, 'ry' + str(ry_index))(self.q_device, wires=q)
                ry_index += 1
                cnot_index += 1
                rx_index += 1


            #Start of final Layer of the Quantum Circuit
            for q in range(0, qubits-1):
                getattr(self, 'cnot' + str(cnot_index))(self.q_device, wires=[q, q + 1])
                getattr(self, 'rx' + str(rx_index))(self.q_device, wires=q)
                getattr(self, 'rx' + str(rx_index))(self.q_device, wires=q)
                getattr(self, 'ry' + str(ry_index))(self.q_device, wires=q)
                ry_index += 1
                cnot_index += 1
                rx_index += 1

    def __init__(self, encoder):
        super().__init__()
        self.n_wires = 8
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.encoder = encoder
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)
        devi = x.device

        out = x
        out2 = torch.zeros(
            x.shape[0], 2 ** self.q_device.n_wires,
            device=x.device)

        for i, t in enumerate(out):
            # Group and encode the data
            qubits_list = self.encoder(self.q_device, t.view(4, 4))

            temp = qubits_list[0].states
            for qubits in qubits_list[1:]:
                temp = torch.kron(temp, qubits.states)

            out2[i] = temp

        states1d = out2
        states1d = states1d.view([out2.shape[0]] + [2] * self.q_device.n_wires)
        self.q_device.states = states1d.type(tq.C_DTYPE)
        self.q_layer(self.q_device)
        x = self.measure(self.q_device)

        if x.shape[0] != bsz:
            x = x.repeat(bsz, 1)

        x = x.reshape(bsz, 4, 2).sum(-1).squeeze()

        x = F.log_softmax(x, dim=1)

        return x


def train(dataflow, model, device, optimizer):
    target_all = []
    output_all = []
    for feed_dict in dataflow['train']:
        inputs = feed_dict['image'].to(device)
        targets = feed_dict['digit'].to(device)
        outputs = model(inputs)
        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        target_all.append(targets)
        output_all.append(outputs)
        print(f"loss: {loss.item()}", end='\r')

    target_all = torch.cat(target_all, dim=0)
    output_all = torch.cat(output_all, dim=0)
    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    train_acc = 100. * corrects / size
    train_loss = F.nll_loss(output_all, target_all).item()

    return train_loss, train_acc


def valid_test(dataflow, split, model, device, qiskit=False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            inputs = feed_dict['image'].to(device)
            targets = feed_dict['digit'].to(device)

            outputs = model(inputs, use_qiskit=qiskit)

            target_all.append(targets)
            output_all.append(outputs)
        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)

    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    test_acc = 100. * corrects / size
    test_loss = F.nll_loss(output_all, target_all).item()
    print(f"{split} set accuracy: {test_acc}")
    print(f"{split} set loss: {test_loss}")
    return test_loss, test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--static', action='store_true', default=True, help='compute with static mode')
    parser.add_argument('--pdb', action='store_true', help='debug with pdb')
    parser.add_argument('--wires-per-block', type=int, default=2, help='wires per block in static mode')
    parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')
    args = parser.parse_args()

    if args.pdb:
        import pdb
        pdb.set_trace()

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset = MNIST(root='./mnist_data', train_valid_split_ratio=[0.9, 0.1], digits_of_interest=[0,1,2,4],
                    n_test_samples=200)
    dataflow = dict()
    print(dataset['train'].n_instance)
    print(dataset['valid'].n_instance)
    print(dataset['test'].n_instance)
    for split in dataset:
        sampler = torch.utils.data.RandomSampler(dataset[split])
        dataflow[split] = torch.utils.data.DataLoader(dataset[split], batch_size=256, sampler=sampler, num_workers=8,
                                                      pin_memory=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    W = H = S = 2
    dupes = [1, 1, 1, 1]
    encoder = SpatialDataEncoder(W, H, S, dupes)
    model = QFCModel(encoder).to(device)

    n_epochs = args.epochs
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    if args.static:
        model.q_layer.static_on(wires_per_block=args.wires_per_block)
    train_loss, train_acc = [], []
    test_loss, test_acc = [], []
    print(device)
    q_layer_circ = tq2qiskit(QFCModel(encoder).q_device, QFCModel(encoder).q_layer)
    q_layer_circ.draw(output='mpl')
    plt.show()
    print('start')
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}:")
        tr_loss, tr_acc = train(dataflow, model, device, optimizer)
        # print("get here?")
        train_loss.append(tr_loss)
        train_acc.append(tr_acc)
        print(optimizer.param_groups[0]['lr'])

        te_loss, te_acc = valid_test(dataflow, 'valid', model, device)
        test_loss.append(te_loss)
        test_acc.append(te_acc)
        scheduler.step()
    print("Done!")
    valid_test(dataflow, 'test', model, device, qiskit=False)

    q_layer_circ = tq2qiskit(QFCModel(encoder).q_device, QFCModel(encoder).q_layer)
    q_layer_circ.draw(output='mpl')
    plt.show()

    plt.figure(figsize=(5, 3))
    plt.plot(range(1, n_epochs + 1), train_acc)
    plt.plot(range(1, n_epochs + 1), test_acc)
    plt.title('Accuracy in {} epochs'.format(n_epochs))

    plt.figure(figsize=(5, 3))
    plt.plot(range(1, n_epochs + 1), train_loss)
    plt.plot(range(1, n_epochs + 1), test_loss)
    plt.title('Loss in {} epochs'.format(n_epochs))
    plt.show()


if __name__ == '__main__':
    main()
