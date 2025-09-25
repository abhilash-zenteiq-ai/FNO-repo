import numpy as np
import gdown
import os
from core.data.fno_dataset import generate_data
from core.data.dataset_eps import generate_data_eps

# ==== Dataloader for FNO_1D ====
def FNOData1D(drive_input, output_file_name):

    # Construct the download URL
    url = drive_input

    # Output path where the file will be saved
    output_path = f'core/data/{output_file_name}'

    # Download
    gdown.download(url, output_path, quiet=False)
    print(f'File saved at {output_path}')

# ==== Dataloader for FNO_2D ====
class FNOData2D:
    def __init__(self, n_samples=1000, size=64):
        self.n_samples = n_samples
        self.size = size

    def load_data(self):
        f, u = generate_data(n_samples=self.n_samples, size=self.size)
        return f[:800], u[:800], f[800:], u[800:]
    
    def load_data_eps(self):
        input_data, output_data = generate_data_eps(n_samples=self.n_samples)

        train_x, test_x = input_data[:1000], input_data[1000:]
        train_y, test_y = output_data[:1000], output_data[1000:]

        val_split = 100
        val_x, val_y = train_x[-val_split:], train_y[-val_split:]
        train_x, train_y = train_x[:-val_split], train_y[:-val_split]

        return train_x, train_y, test_x, test_y, val_x, val_y
