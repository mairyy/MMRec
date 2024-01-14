import pickle
import copy
import numpy as np
from scipy.sparse import csr_matrix

def read_binary_matrix_with_pickle(file_path):
    with open(file_path, 'rb') as file:
        # Sử dụng pickle để đọc dữ liệu từ file nhị phân
        matrix = pickle.load(file)

    return matrix

def add_one_to_matrix_elements(matrix):
    # Cộng thêm 1 đơn vị cho mỗi phần tử trong ma trận
    result_matrix = [[element + 1 for element in row] for row in matrix]

    return result_matrix

def write_binary_matrix_with_pickle(matrix, file_path):
    with open(file_path, 'wb') as file:
        # Sử dụng pickle để ghi dữ liệu vào file nhị phân
        pickle.dump(matrix, file)

file = 'datasets/Beauty/neg'
# Đọc ma trận từ file nhị phân bằng pickle
binary_matrix = read_binary_matrix_with_pickle(file)

# for i in range(10):
#     print(binary_matrix[i])

# Cộng thêm 1 đơn vị cho mỗi phần tử trong ma trận
# result_matrix = add_one_to_matrix_elements(binary_matrix)

# Ghi ma trận kết quả vào một file mới
# write_binary_matrix_with_pickle(result_matrix, file)

# Hiển thị ma trận kết quả

# print(binary_matrix)

abc = set()
for i in range(len(binary_matrix)):
    for j in range(len(binary_matrix[i])):
        abc.add(binary_matrix[i][j])
        if(binary_matrix[i][j] == 1):
            print(111)

print(len(abc))

def construct_graphs(num_items=12102, distance=3, path='datasets/Beauty/'):
    with open(path + 'seq', 'rb') as fs:
        seqs = pickle.load(fs)
    user = list()
    r, c, d = list(), list(), list()
    for i, seq in enumerate(seqs):
        print(f"Processing {i}/{len(seqs)}          ", end='\r')
        for dist in range(1, distance + 1):
            if dist >= len(seq): break;
            r += copy.deepcopy(seq[+dist:])
            c += copy.deepcopy(seq[:-dist])
            r += copy.deepcopy(seq[:-dist])
            c += copy.deepcopy(seq[+dist:])
    d = np.ones_like(r)
    iigraph = csr_matrix((d, (r, c)), shape=(num_items, num_items))
    print('Constructed i-i graph, density=%.6f' % (len(d) / (num_items ** 2)))
    with open('datasets/Beauty/trn1', 'wb') as fs:
        pickle.dump(iigraph, fs)

# construct_graphs()
