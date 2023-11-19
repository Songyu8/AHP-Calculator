import numpy as np
class AHP:
    """
    相关信息的传入和准备
    """

    def __init__(self, array):
        ## 记录矩阵相关信息
        self.array = array
        ## 记录矩阵大小
        self.n = array.shape[0]
        # 初始化RI值，用于一致性检验
        self.RI_list = [0,0,0,0.58,0.90,1.12]
        
        self.weight=self.cal_weight_by_arithmetic_method()

        self.eig_vector=np.matmul(array,self.weight)

        self.nor_eigvector=self.eig_vector/self.weight

        # 矩阵的Lamda
        self.Lamda = np.mean(self.nor_eigvector)

        # 矩阵的一致性指标CI
        self.CI_val = (self.Lamda - self.n) / (self.n - 1)
        # 矩阵的一致性比例CR
        self.CR_val = self.CI_val / (self.RI_list[self.n])

    """
    一致性判断
    """

    def test_consist(self):
        # 打印矩阵的一致性指标CI和一致性比例CR
        print("判断矩阵的CI值为：" + str(self.CI_val))
        print("判断矩阵的CR值为：" + str(self.CR_val))
        # 进行一致性检验判断
        if self.n == 2:  # 当只有两个子因素的情况
            print("仅包含两个子因素，不存在一致性问题")
        else:
            if self.CR_val < 0.1:  # CR值小于0.1，可以通过一致性检验
                print("判断矩阵的CR值为" + str(self.CR_val) + ",通过一致性检验")
                return True
            else:  # CR值大于0.1, 一致性检验不通过
                print("判断矩阵的CR值为" + str(self.CR_val) + "未通过一致性检验")
                return False

    """
    算术平均法求权重
    """

    def cal_weight_by_arithmetic_method(self):
        # 求矩阵的每列的和
        col_sum = np.sum(self.array, axis=0)
        # 将判断矩阵按照列归一化
        array_normed = self.array / col_sum
        # 计算权重向量
        array_weight = np.sum(array_normed, axis=1) / self.n
        # 返回权重向量的值
        return array_weight

if __name__ == "__main__":
    # 给出判断矩阵
    b = np.array([[1, 5, 2,3], [1/5, 1, 1/4,1/6], [1/2, 4,1, 1/2],[1/3,6,2,1]])
    # b=np.array([[1,3,1/4],[1/3,1,1/6],[4,6,1]])
    weight1 = AHP(b).cal_weight_by_arithmetic_method()
    r=np.array([[1, 2, 3,3], [1/2, 1, 5,1/2], [1/3, 1/5,1, 1/4],[1/3,2,4,1]])
    p=np.array([[1, 1/2, 1/3,1/4], [2, 1, 1/3,1/2], [3, 3,1, 4],[4,2,1/4,1]])
    l=np.array([[1, 4, 3,1/4], [1/4, 1, 2,1/3], [1/3, 1/2,1, 4],[4,3,1/4,1]])
    f=np.array([[1, 5, 3,1/2], [1/5, 1, 3,1/3], [1/3, 1/3,1, 1/4],[2,3,4,1]])
    AHP(b).test_consist()
    print("Lamba:\n",AHP(b).Lamda)
    # 打印权重向量
    print("权重向量为：\n", weight1)
    r_weight=AHP(r).cal_weight_by_arithmetic_method()
    p_weight=AHP(p).cal_weight_by_arithmetic_method()
    l_weight=AHP(l).cal_weight_by_arithmetic_method()
    f_weight=AHP(f).cal_weight_by_arithmetic_method()
    print(r_weight.reshape(r_weight.shape[0],1))

    result_horizontal = np.concatenate((r_weight.reshape(r_weight.shape[0],1), p_weight.reshape(p_weight.shape[0],1), l_weight.reshape(l_weight.shape[0],1), f_weight.reshape(f_weight.shape[0],1)),axis=1)
    print(result_horizontal)
    composite_priority=np.matmul(result_horizontal,weight1)

    print(composite_priority)
    # print(weight1)

