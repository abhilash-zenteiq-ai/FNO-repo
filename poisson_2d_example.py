import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from core.models.fno_2d.fno2d_model import FNO2D
from core.data.fnodataloader import FNOData2D
import warnings

warnings.filterwarnings('ignore')			
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'			
tf.get_logger().setLevel('ERROR')


class FNOTrainer:
    def __init__(self, epochs=10, batch_size=20, output_dir="output/fno_2d/exp4"):
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.model = FNO2D()
        self.model.compile(optimizer='adam', loss='mse')

        self.train_f, self.train_u, self.test_f, self.test_u = FNOData2D().load_data()
        self.train_losses = []

    def train(self):
        for epoch in range(self.epochs):
            idx = np.random.permutation(len(self.train_f))
            f_shuffled = self.train_f[idx]
            u_shuffled = self.train_u[idx]

            epoch_loss = 0
            for i in range(0, len(f_shuffled), self.batch_size):
                x_batch = f_shuffled[i:i + self.batch_size]
                y_batch = u_shuffled[i:i + self.batch_size]
                loss = self.model.train_on_batch(x_batch, y_batch)
                epoch_loss += loss

            avg_loss = epoch_loss / (len(self.train_f) // self.batch_size)
            self.train_losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.6f}")

        self._plot_loss()

    def _plot_loss(self):
        plt.figure()
        plt.plot(range(1, self.epochs + 1), self.train_losses, marker='o')
        plt.title("Epoch vs. Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.grid(True)
        path = os.path.join(self.output_dir, "loss_curve.png")
        plt.savefig(path)
        print(f"Saved loss curve at: {path}")

    def evaluate(self):
        pred = self.model.predict(self.test_f[:1])[0, ..., 0]
        true = self.test_u[0, ..., 0]
        error = np.abs(pred - true)

        # === Error Metrics ===
        max_error = np.max(error)
        mean_abs_error = np.mean(error)
        l2_error = np.linalg.norm(pred - true)
        relative_error = l2_error / np.linalg.norm(true)

        # === Print & Save Errors ===
        print(f"Max Error: {max_error:.6f}")
        print(f"Mean Absolute Error: {mean_abs_error:.6f}")
        print(f"L2 Error: {l2_error:.6f}")
        print(f"Relative L2 Error: {relative_error:.6f}")

        metrics_path = os.path.join(self.output_dir, "metrics.txt")
        with open(metrics_path, "w") as f:
            f.write(f"Max Error: {max_error:.6f}\n")
            f.write(f"Mean Absolute Error: {mean_abs_error:.6f}\n")
            f.write(f"L2 Error: {l2_error:.6f}\n")
            f.write(f"Relative L2 Error: {relative_error:.6f}\n")
        print(f"Saved metrics at: {metrics_path}")

        # === Plot Prediction vs Ground Truth ===
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title("Predicted u")
        plt.imshow(pred, cmap='viridis')
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.title("True u")
        plt.imshow(true, cmap='viridis')
        plt.colorbar()
        plt.tight_layout()
        path_pred = os.path.join(self.output_dir, "pred_vs_true.png")
        plt.savefig(path_pred)
        print(f"Saved plot at: {path_pred}")

        # === Error Plot ===
        plt.figure()
        plt.imshow(error, cmap='Reds')
        plt.title("Absolute Error |u_pred - u_true|")
        plt.colorbar()
        plt.tight_layout()
        path_error = os.path.join(self.output_dir, "error_plot.png")
        plt.savefig(path_error)
        print(f"Saved error plot at: {path_error}")


if __name__ == "__main__":
    trainer = FNOTrainer()
    trainer.train()
    trainer.evaluate()
