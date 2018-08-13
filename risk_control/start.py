import pandas as pd
import numpy as np
from dataclean import DataClean
from models import Models
from model_fusion import Blend


def clean_data():
    df_train = pd.read_csv('input/LoanStats_2016Q3.csv', skiprows=1, low_memory=False)
    df_test = pd.read_csv('input/LoanStats_2016Q4.csv', skiprows=1, low_memory=False)
    cleaner_train = DataClean(df_train)
    cleaner_test = DataClean(df_test)
    cleaner_train.clean_y()  # 先clean_y再clean_x，否则loan_status这一列就会被dummy掉
    cleaner_train.clean_x()
    cleaner_train.drop_cor()
    cleaner_test.clean_y()
    cleaner_test.clean_x()
    cleaner_test.drop_cor()
    y_train = pd.DataFrame()
    y_test = pd.DataFrame()
    y_train['y'] = pd.Series(cleaner_train.df.loan_status)
    x_train = cleaner_train.df.drop('loan_status', 1, inplace=False)
    y_test['y'] = pd.Series(cleaner_test.df.loan_status)
    x_test = cleaner_test.df.drop('loan_status', 1, inplace=False)
    print('开始生成数据'.center(50, '*'))
    x_train.to_csv('cleaned_data/cleaned_x_train.csv')
    print('cleaned_x_train完成')
    y_train.to_csv('cleaned_data/cleaned_y_train.csv')
    print('cleaned_y_train完成')
    x_test.to_csv('cleaned_data/cleaned_x_test.csv')
    print('cleaned_x_test完成')
    y_test.to_csv('cleaned_data/cleaned_y_test.csv')
    print('cleaned_y_test完成')
    print('数据生成完毕'.center(50, '*'))


def get_result():
    # ---------------------------------------读取数据-----------------------------------------
    x_train = pd.read_csv('cleaned_data/cleaned_x_train.csv')
    x_test = pd.read_csv('cleaned_data/cleaned_x_test.csv')
    y_train = pd.read_csv('cleaned_data/cleaned_y_train.csv')
    y_test = pd.read_csv('cleaned_data/cleaned_y_test.csv')

    blender = Blend(x_train, x_test, y_train, y_test)
    blender.blending()
    scores = blender.score()
    print(scores)
    prediction = pd.DataFrame()
    prediction['y_pred'] = blender.prediction()
    prediction.to_csv('output/prediction.csv')


func_dict = {
    '1': clean_data,
    '2': get_result
}

if __name__ == '__main__':
    while True:
        print("""
        1.清洗数据
        2.进行模型融合并预测结果
        """)
        choice = input('请输入要进行的操作:').strip()
        if choice in func_dict:
            func_dict[choice]()
        if choice == 'q':
            break
