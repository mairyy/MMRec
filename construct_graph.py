import pickle
import numpy as np
from scipy.sparse import csr_matrix
import torch as t
import scipy.sparse as sp


def read_binary_matrix_with_pickle(file_path):
    with open(file_path, 'rb') as file:
        # Sử dụng pickle để đọc dữ liệu từ file nhị phân
        matrix = pickle.load(file)

    return matrix

def get_knn_adj_mat(mm_embedding, knn_k, device):
#     print(mm_embedding[25])
    mask_eged = set()
    embedding_norm = t.norm(mm_embedding, p=2, dim=-1, keepdim=True)

    # Tạo mặt nạ cho các item có embedding khác 0
    mask = (embedding_norm != 0).float()
    for i in range(len(mask)):
        if(mask[i] == 0):
            mask_eged.add(i)

    # Chia từng phần tử của mm_embedding cho embedding_norm (tránh chia cho 0)
    context_norm = mm_embedding.div(embedding_norm + 1e-8)

    # Áp dụng mặt nạ để chỉ tính sim cho các item có embedding khác 0
    context_norm *= mask

    # Chuyển context_norm lên GPU
    context_norm = context_norm.to(device)

    sim = t.mm(context_norm, context_norm.transpose(1, 0))
        # Đặt giá trị của đường chéo bằng 0
    sim.fill_diagonal_(0)
    abc, knn_ind = t.topk(sim, knn_k, dim=-1)

    adj_size = sim.size()
    del sim



    # construct sparse adj
    indices0 = t.arange(knn_ind.shape[0]).to(device)
    indices0 = t.unsqueeze(indices0, 1)
    indices0 = indices0.expand(-1, knn_k)
    indices = t.stack((t.flatten(indices0), t.flatten(knn_ind)), 0)
    # Chuyển indices về CPU
    indices = indices.cpu()
    
        # Remove indices corresponding to zero embeddings
    mask_indices = mask.squeeze().nonzero().squeeze()
    mask_indices_set = set(mask_indices.tolist())
    indices_list = indices.tolist()
    indices_filtered = [[i, j] for i, j in zip(*indices_list) if i in mask_indices_set]
    indices = t.tensor(indices_filtered).t()

    return indices

def tensor_to_csr_matrix(indices, shape):
    # Gán giá trị 1 cho mỗi phần tử của ma trận CSR
    values = np.ones(indices.shape[1])
    csr_matrix_result = csr_matrix((values, (indices[0]+1, indices[1]+1)), shape=(12102, 12102))
    return csr_matrix_result


# Đọc ma trận từ file nhị phân bằng pickle
binary_matrix = read_binary_matrix_with_pickle('./datasets/Beauty/vision')
print(binary_matrix.shape)

# Tạo csr_matrix từ ma trận nhúng
# Chuyển đổi một số phép toán lên GPU
device = 'cuda' if t.cuda.is_available() else 'cpu'
csr_matrix_result = get_knn_adj_mat(binary_matrix, knn_k=20, device=device)
# for i in range(len(csr_matrix_result[0])):
#     if(csr_matrix_result[0][i] == 27):
#         print(csr_matrix_result[0][i])
#         print(csr_matrix_result[1][i])
print(csr_matrix_result)

arr = tensor_to_csr_matrix(csr_matrix_result, binary_matrix.shape)
print(arr)
# with open("/kaggle/working/adj_vision", 'wb') as fs:
#     pickle.dump(arr, fs)

