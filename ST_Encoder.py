import torch
import torchquantum as tq

def custom_amplitude_encode(state, data, normalize=True):
    """
    Custom amplitude encoding function to encode classical data into quantum state amplitudes.

    Args:
        state (torch.Tensor): Quantum state tensor to be modified.
        data (torch.Tensor): Classical data to encode.
        normalize (bool): Whether to normalize the data.

    Returns:
        torch.Tensor: Encoded quantum state.
    """
    if normalize:
        data = data / torch.norm(data)
    if state.size(0) != data.size(0):
        raise ValueError("Size of the state tensor and data tensor must match.")
    state[:] = data
    print("Encoded Data being put into large tensor:", state)
    return state

# Define the data grouping function
def group_data(data, W, H, S):
    """
    Group data into smaller groups based on hyperparameters (W, H, S)
    W: Width of the group
    H: Height of the group
    S: Step size for the filter
    """
    groups = []
    for i in range(0, data.shape[0] - H + 1, S):
        for j in range(0, data.shape[1] - W + 1, S):
            group = data[i:i+H, j:j+W]
            groups.append(group)
    #print(groups)
    return groups

def duped_qubits(qubits, dups):
    maxs = []

    for qubit, factor in zip(qubits, dups):
        duped_qubit = qubit
        for i in range(factor - 1):
            #Commits the kronecker product of itself to do the nonlinear duplication
            duped_qubit = torch.kron(duped_qubit, qubit)
        #puts the duplication into the output tensor to keep in same form as grouped data
        maxs.append(duped_qubit)
    #print(maxs)
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
        print("Grouped Data:", self.groups)
        
        output_qubits = duped_qubits(self.groups, self.duper)
        print("Duped Data:", output_qubits)
        # Apply quantum encoding to each group
        qubits_list = []
        for idx, group in enumerate(output_qubits):
            # Flatten the group data for amplitude encoding
            flattened_group = group.flatten()
            num_qubits = int(torch.ceil(torch.log2(torch.tensor(flattened_group.size(0))).float()).item())
            #print(f"Group {idx} requires {num_qubits} qubits.")
            # Create quantum state
            qubits = tq.QuantumDevice(n_wires=num_qubits)
            #print(qubits)
            # Initialize the quantum state tensor to the correct size
            qubits.states = torch.zeros(2**num_qubits, dtype=torch.cfloat)
            #print(f"Initial state tensor size: {qubits.states.size()}")
            # Amplitude encoding
            custom_amplitude_encode(qubits.states, flattened_group)
            qubits_list.append(qubits)
        
        return qubits_list

def main():
    qdev = tq.QuantumDevice(n_wires=8)
    data = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                         [0.5, 0.6, 0.7, 0.8],
                         [0.9, 1.0, 1.1, 1.2],
                         [1.3, 1.4, 1.5, 1.6]])

    # Parameters for the encoder
    W = int(input("Input value for width: "))
    H = int(input("Input value for height: "))
    S = int(input("Input value for stride: "))
    
    dupes = []
        
    for i in range(0, 4):
        ele = int(input("Give a value for element duplication: "))
        # adding the element
        dupes.append(ele)    
    
    # Create the spatial data encoder
    encoder = SpatialDataEncoder(W, H, S, dupes)

    # Forward pass (encoding without measuring)
    qubits_list = encoder(qdev, data)
    
    
    for idx, qubits in enumerate(qubits_list):
        print(f"Qubits {idx}:")
        print(f"Shape of states: {qubits.states.shape}")
        # Avoid printing the whole qubits object which triggers the __repr__ method causing the error
        print(f"Qubits info - number of qubits: {qubits.n_wires}, state tensor size: {qubits.states.size()}")

if __name__ == "__main__":
    main()
