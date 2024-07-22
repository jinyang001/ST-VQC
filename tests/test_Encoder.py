import torch
import torchquantum as tq
from ST_Encoder import SpatialDataEncoder

def main():
    data = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                         [0.5, 0.6, 0.7, 0.8],
                         [0.9, 1.0, 1.1, 1.2],
                         [1.3, 1.4, 1.5, 1.6]])

    larger_data = torch.tensor([[0.1 * (i + j) for j in range(16)] for i in range(16)])
    q_device = tq.QuantumDevice(n_wires=8)
    # Parameters for the encoder
    for i in [2**n for n in range(1, 6)]:
        W = H = S = i

        # Select the appropriate data based on the value of i
        if i > 2:
            current_data = larger_data
        else:
            current_data = data

        print(f"Testing with W=H=S={i}, data size={current_data.size()}")

        # Create the spatial data encoder
        encoder = SpatialDataEncoder(W, H, S, [1, 1, 1, 1])  # Assume no duplication for simplicity
        try:
            # Forward pass (encoding without measuring)
            qubits_list = encoder(q_device, current_data)

            for idx, qubits in enumerate(qubits_list):
                print(f"Qubits {idx}:")
                print(f"Shape of states: {qubits.states.shape}")
                # Avoid printing the whole qubits object which triggers the __repr method causing the error
                print(f"Qubits info - number of qubits: {qubits.n_wires}, state tensor size: {qubits.states.size()}")
        except IndexError:
            print(f"Failed to encode with W=H=S={i}, due to insufficient data grouping.")

if __name__ == "__main__":
    main()
