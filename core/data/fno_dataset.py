import numpy as np

def generate_data(n_samples=10, size=64, m_max=5, n_max=5):
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)

    f_list, u_list = [], []

    for _ in range(n_samples):
        m = np.random.randint(1, m_max + 1)
        n = np.random.randint(1, n_max + 1)

        f = np.sin(m * np.pi * X) * np.sin(n * np.pi * Y)
        u = f / (np.pi**2 * (m**2 + n**2))  # Exact Poisson solution

        f_list.append(f[..., None])  # shape: [H, W, 1]
        u_list.append(u[..., None])

    return np.array(f_list).astype(np.float32), np.array(u_list).astype(np.float32)



    '''f_list.append(f)  # shape: [H, W, 1]
    u_list.append(u)

    f_array = np.array(f_list)
    u_array = np.array(u_list)

    mesh_x = np.repeat(X[np.newaxis, :, :], n_samples, axis=0)
    mesh_y = np.repeat(Y[np.newaxis, :, :], n_samples, axis=0)

    input_tensor = np.stack([f_array, mesh_x, mesh_y], axis=-1).astype(np.float32)
    output_tensor = u_array[:, np.newaxis, :, :].astype(np.float32)

    return input_tensor, output_tensor'''
