import numpy as np

def main():
    with open('./robot2cameraMartix.txt','r') as f:
        content = f.read().replace('\n','')
        matrix_list = eval(content)
        matrix_arr = np.array(matrix_list)
        
        # 将矩阵求逆
        matrix_inv = np.linalg.inv(matrix_arr)
        
        # 修改为易于json识别的格式
        with open('./matrix_world2robot','w') as fw:
            for index_line in range(len(matrix_inv)-1):
                fw.write(f'{list(matrix_inv[index_line])},\n')
            fw.write('[0,0,0,1]')
        print('matrix of world2robot written to ./matrix_world2robot!!!')

if __name__ == '__main__':
    main()