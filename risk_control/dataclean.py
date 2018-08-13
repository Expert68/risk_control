import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# --------------------------定义数据清理类-----------------------
class DataClean:
    def __init__(self, df):
        self.df = df

    def clean_x(self):
        self.df.drop('id', 1, inplace=True)
        self.df.drop('member_id', 1, inplace=True)
        self.df.int_rate = pd.Series(self.df.int_rate).str.replace('%', '').astype(float)
        self.df.dropna(axis=0, how='all', inplace=True)
        self.df.dropna(axis=1, how='all', inplace=True)
        self.df.drop(['emp_title'], 1, inplace=True)
        self.df.replace('n/a', np.nan, inplace=True)
        self.df.emp_length.fillna(value=0, inplace=True)
        self.df['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
        self.df['emp_length'] = self.df['emp_length'].astype(int)
        self.df.revol_util = pd.Series(self.df.revol_util).str.replace('%', '').astype(float)
        self.df.drop('desc', 1, inplace=True)
        self.df.drop('verification_status_joint', 1, inplace=True)
        self.df.drop('zip_code', 1, inplace=True)
        self.df.drop('addr_state', 1, inplace=True)
        self.df.drop('earliest_cr_line', 1, inplace=True)
        self.df.drop('revol_util', 1, inplace=True)
        self.df.drop('purpose', 1, inplace=True)
        self.df.drop('title', 1, inplace=True)
        self.df.drop('term', 1, inplace=True)
        self.df.drop('issue_d', 1, inplace=True)
        self.df.drop(['out_prncp', 'out_prncp_inv', 'total_pymnt',
                      'total_pymnt_inv', 'total_rec_prncp', 'grade', 'sub_grade'], 1, inplace=True)
        self.df.drop(['total_rec_int', 'total_rec_late_fee',
                      'recoveries', 'collection_recovery_fee',
                      'collection_recovery_fee'], 1, inplace=True)
        self.df.drop(['last_pymnt_d', 'last_pymnt_amnt',
                      'next_pymnt_d', 'last_credit_pull_d'], 1, inplace=True)
        self.df.drop(['policy_code'], 1, inplace=True)
        self.df.drop('annual_inc_joint', 1, inplace=True)
        self.df.drop('dti_joint', 1, inplace=True)
        self.df.fillna(0, inplace=True)
        self.df = pd.get_dummies(self.df)

    def clean_y(self):
        self.df.loan_status.replace('Fully Paid', int(1), inplace=True)
        self.df.loan_status.replace('Current', int(1), inplace=True)
        self.df.loan_status.replace('Late (16-30 days)', int(0), inplace=True)
        self.df.loan_status.replace('Late (31-120 days)', int(0), inplace=True)
        self.df.loan_status.replace('Charged Off', np.nan, inplace=True)
        self.df.loan_status.replace('In Grace Period', np.nan, inplace=True)
        self.df.loan_status.replace('Default', np.nan, inplace=True)
        self.df.loan_status.dropna(inplace=True)

    def cor(self):
        cor = self.df.corr()
        cor.loc[:, :] = np.tril(cor, k=-1)
        cor = cor.stack()
        return cor[(cor > 0.55) | (cor < -0.55)]

    def drop_cor(self):
        self.df.drop(['funded_amnt', 'funded_amnt_inv', 'installment'], axis=1, inplace=True)
