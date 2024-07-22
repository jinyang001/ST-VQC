import torch
import torchquantum as tq

# Custom amplitude encode function based on StateEncoder method
def custom_amplitude_encode(state, data, normalize=True):
    if normalize:
        data = data / (torch.sqrt((data.abs() ** 2).sum())).unsqueeze(-1)
    # Pad with zeros to match the state vector size
    data_padded = torch.cat(
        (data, torch.zeros(2**int(torch.log2(torch.tensor(state.shape[0])).item()) - data.shape[0], device=data.device)),
        dim=-1
    )
    state[:] = data_padded.type(state.dtype)
    return state

# Define the data grouping function
def group_data(data, W, H, S):
    groups = []
    for i in range(0, data.shape[0] - H + 1, S):
        for j in range(0, data.shape[1] - W + 1, S):
            group = data[i:i+H, j:j+W]
            groups.append(group)
    return groups

def duped_qubits(qubits, dups):
    maxs = []
    for qubit, factor in zip(qubits, dups):
        duped_qubit = qubit
        for i in range(factor - 1):  # we loop through the groups that have been established
            duped_qubit = torch.kron(duped_qubit, qubit)  # then we do the Kronecker product n many times
        maxs.append(duped_qubit)
    return maxs

# Define the quantum encoding module
class SpatialDataEncoder(tq.QuantumModule):
    def __init__(self, W, H, S, duper):
        super().__init__()
        self.W = W
        self.H = H
        self.S = S
        self.duper = duper
        self.groups = None

    def forward(self, q_device: tq.QuantumDevice, x):
        # Group the data
        self.groups = group_data(x, self.W, self.H, self.S)
        #print("Grouped Data:", self.groups)

        output_qubits = duped_qubits(self.groups, self.duper)

        # Apply quantum encoding to each group
        qubits_list = []
        for idx, group in enumerate(output_qubits):
            # Flatten the group data for amplitude encoding
            flattened_group = group.flatten()
            num_qubits = int(torch.ceil(torch.log2(torch.tensor(flattened_group.size(0))).float()).item())
            qubits = tq.QuantumDevice(n_wires=num_qubits)
            #print(qubits)
            qubits.states = torch.zeros(2**num_qubits, dtype=torch.cfloat)
            custom_amplitude_encode(qubits.states, flattened_group)
            qubits_list.append(qubits)

        return qubits_list

def main():
    data = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                         [0.5,   0.6, 0.7, 0.8],
                         [0.9, 1.0, 1.1, 1.2],
                         [1.3, 1.4, 1.5, 1.6]])

    # Parameters for the encoder
    W = 2
    H = 2
    S = 2

    dupes = [1,1,1,1]
    
    # Create the spatial data encoder
    encoder = SpatialDataEncoder(W, H, S, dupes)

    # Forward pass (encoding without measuring)
    qubits_list = encoder(tq.QuantumDevice(n_wires=4), data)

    for idx, qubits in enumerate(qubits_list):
        print(f"Qubits {idx}:")
        print(f"Shape of states: {qubits.states.shape}")
        # Avoid printing the whole qubits object which triggers the __repr__ method causing the error
        print(f"Qubits info - number of qubits: {qubits.n_wires}, state tensor size: {qubits.states.size()}")

if __name__ == "__main__":
    main()
