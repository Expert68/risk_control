# risk_control
应用于p2p公司，用于判断借贷人的风险程度，判断借贷人能否及时还欠款


### 金融借贷风控
数据来源：lending club 2016年Q3和Q4数据：https://www.lendingclub.com/info/download-data.action


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
df = pd.read_csv('input/LoanStats_2016Q3.csv',skiprows=1,low_memory=False)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>member_id</th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>...</th>
      <th>hardship_payoff_balance_amount</th>
      <th>hardship_last_payment_amount</th>
      <th>disbursement_method</th>
      <th>debt_settlement_flag</th>
      <th>debt_settlement_flag_date</th>
      <th>settlement_status</th>
      <th>settlement_date</th>
      <th>settlement_amount</th>
      <th>settlement_percentage</th>
      <th>settlement_term</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>30000.0</td>
      <td>30000.0</td>
      <td>30000.0</td>
      <td>60 months</td>
      <td>13.99%</td>
      <td>697.90</td>
      <td>C</td>
      <td>C3</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cash</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>20150.0</td>
      <td>20150.0</td>
      <td>20150.0</td>
      <td>60 months</td>
      <td>24.99%</td>
      <td>591.32</td>
      <td>E</td>
      <td>E4</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cash</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>30000.0</td>
      <td>30000.0</td>
      <td>30000.0</td>
      <td>36 months</td>
      <td>10.99%</td>
      <td>982.02</td>
      <td>B</td>
      <td>B4</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cash</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>15000.0</td>
      <td>15000.0</td>
      <td>15000.0</td>
      <td>36 months</td>
      <td>13.99%</td>
      <td>512.60</td>
      <td>C</td>
      <td>C3</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cash</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>18000.0</td>
      <td>18000.0</td>
      <td>18000.0</td>
      <td>60 months</td>
      <td>14.49%</td>
      <td>423.42</td>
      <td>C</td>
      <td>C4</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cash</td>
      <td>N</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 145 columns</p>
</div>



### 获取列名，对一些不重要的或具有干扰意义的列进行直接删除


```python
df.columns
```




    Index(['id', 'member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv',
           'term', 'int_rate', 'installment', 'grade', 'sub_grade',
           ...
           'hardship_payoff_balance_amount', 'hardship_last_payment_amount',
           'disbursement_method', 'debt_settlement_flag',
           'debt_settlement_flag_date', 'settlement_status', 'settlement_date',
           'settlement_amount', 'settlement_percentage', 'settlement_term'],
          dtype='object', length=145)




```python
df.drop(['id','member_id','zip_code','addr_state'],1,inplace=True)
```


```python
df.int_rate = pd.Series(df.int_rate).str.replace('%', '').astype(float)
```

### 将所有空行删除


```python
df.dropna(axis=0,how='all',inplace=True)
```

### 重点特征关注
1. load_amnt和funded_amnt
2. emp_title：职位信息
3. emp_length: 工作年限
4. verification_status: 收入信息是否被确认
5. loan_status: 贷款情况，即最后的判断结果

###  loan_amnt 和 funded_amnt


```python
df.loan_amnt[pd.isnull(df.loan_amnt)]
```




    Series([], Name: loan_amnt, dtype: float64)




```python
df.query('loan_amnt != funded_amnt')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>emp_title</th>
      <th>emp_length</th>
      <th>...</th>
      <th>hardship_payoff_balance_amount</th>
      <th>hardship_last_payment_amount</th>
      <th>disbursement_method</th>
      <th>debt_settlement_flag</th>
      <th>debt_settlement_flag_date</th>
      <th>settlement_status</th>
      <th>settlement_date</th>
      <th>settlement_amount</th>
      <th>settlement_percentage</th>
      <th>settlement_term</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
<p>0 rows × 141 columns</p>
</div>




```python
df.loan_amnt.value_counts()
```




    10000.0    7328
    12000.0    5358
    20000.0    5140
    15000.0    5082
    5000.0     4429
    8000.0     3924
    6000.0     3729
    35000.0    3306
    16000.0    2934
    24000.0    2849
    30000.0    2365
    25000.0    2240
    18000.0    2199
    7000.0     1893
    14000.0    1694
    4000.0     1532
    3000.0     1496
    9000.0     1249
    21000.0    1160
    28000.0    1152
    9600.0      928
    2000.0      884
    40000.0     878
    7200.0      813
    13000.0     733
    11000.0     732
    11200.0     714
    8400.0      674
    4800.0      670
    14400.0     660
               ... 
    32325.0       1
    24925.0       1
    33425.0       1
    35750.0       1
    36600.0       1
    24725.0       1
    31100.0       1
    33725.0       1
    33075.0       1
    34275.0       1
    30550.0       1
    38725.0       1
    25025.0       1
    38800.0       1
    36750.0       1
    35300.0       1
    36625.0       1
    30325.0       1
    36675.0       1
    38625.0       1
    35200.0       1
    35050.0       1
    30750.0       1
    36200.0       1
    36950.0       1
    39750.0       1
    33225.0       1
    29075.0       1
    39725.0       1
    39075.0       1
    Name: loan_amnt, Length: 1411, dtype: int64



### emp_title: 职位信息


```python
df.emp_title.value_counts()
```




    Teacher                                     1931
    Manager                                     1701
    Owner                                        990
    Supervisor                                   785
    Driver                                       756
    Registered Nurse                             752
    RN                                           731
    Sales                                        664
    Project Manager                              526
    General Manager                              483
    Office Manager                               466
    Director                                     415
    owner                                        384
    Engineer                                     382
    President                                    351
    manager                                      314
    Operations Manager                           314
    Vice President                               288
    Nurse                                        284
    teacher                                      284
    Attorney                                     275
    Accountant                                   274
    Sales Manager                                263
    Analyst                                      246
    Administrative Assistant                     243
    Police Officer                               230
    driver                                       222
    Account Manager                              213
    Technician                                   211
    Executive Assistant                          205
                                                ... 
    Bartender/Production floor tech                1
    Massage Therapist/server                       1
    Financial/Insurance Coordinator                1
    Paint Tech Team Lead                           1
    Delivery supervisor                            1
    Car Maintainer                                 1
    compliance tester                              1
    RN. Clinical Quality Manager                   1
    Retail Marketing Designer                      1
    MANAGER, MEMBER SERVICES                       1
    Clinical Nurse Lead                            1
    Loan Coordinator                               1
    EHR Project Manager                            1
    Database Specialist                            1
    Consumer Loan Officer                          1
    Assistant Administrator - Human Resource       1
    Sr. Web Developer                              1
    Asst sales manager                             1
    Engineering Tech 3                             1
    WARRANTY ADMIN                                 1
    laborer assembly                               1
    Logistics Operations Manager                   1
    Healthcare technology consultant               1
    Clinical Trial Specialist II                   1
    Union Autoworker                               1
    Adult Trauma Coordinator RN                    1
    ELIG CLERK                                     1
    Driver/ firefighter                            1
    Marina Manager                                 1
    CDL Licensing trainer/Driver                   1
    Name: emp_title, Length: 37420, dtype: int64




```python
df.emp_title.head()
```




    0          General Manager
    1                   Server
    2                   server
    3          Fiscal Director
    4     utility technician 2
    Name: emp_title, dtype: object




```python
# 职位名称太多太杂，所以删掉这一列，防止对结果产生干扰
df.drop('emp_title',1,inplace=True)
```

### emp_length: 工作年限属于重要特征，所以需要保留，一个人的工作年限越长，一般也意味着这个人的贷款信誉越好


```python
df.emp_length.value_counts()
```




    10+ years    34219
    2 years       9066
    3 years       7925
    < 1 year      7104
    1 year        6991
    5 years       6170
    4 years       6022
    6 years       4406
    8 years       4168
    9 years       3922
    7 years       3205
    Name: emp_length, dtype: int64




```python
# 进行正则替换
df.replace('n/a',np.nan,inplace=True)
df['emp_length'].fillna(value=0,inplace=True)
df['emp_length'].replace(to_replace='[^0-9]+',value='',inplace=True,regex=True)
df['emp_length'] = df['emp_length'].astype(int)
```


```python
df.emp_length.value_counts()
```




    10    34219
    1     14095
    2      9066
    3      7925
    5      6170
    4      6022
    0      5922
    6      4406
    8      4168
    9      3922
    7      3205
    Name: emp_length, dtype: int64



### verification_status:收入信息确认  收入信息确认是能够确认贷款人是否诚信的重要标志，所以对于该列信息需要进行研究


```python
df.verification_status.value_counts()
```




    Source Verified    42253
    Verified           31356
    Not Verified       25511
    Name: verification_status, dtype: int64



### Loan_status:计算目标  Loan_status作为计算目标，需要对这列进行研究，并转化为比较明确的数值型或者类别型标签，从而将问题转化为分类问题或者是回归问题


```python
df.loan_status.value_counts()
```




    Current               52061
    Fully Paid            32435
    Charged Off           11317
    Late (31-120 days)     1769
    In Grace Period         928
    Late (16-30 days)       505
    Default                 105
    Name: loan_status, dtype: int64




```python
pd.unique(df.loan_status.values.ravel())
```




    array(['Fully Paid', 'Current', 'Charged Off', 'Late (31-120 days)',
           'In Grace Period', 'Late (16-30 days)', 'Default'], dtype=object)



- current:代表还在还款中
- fully paid代表还款完成
- 其他的都是逾期用户，属于信誉不好的用户

所以通过replace函数，将这一列改为0和1的变量，让问题变成二分类问题


```python
df.loan_status.replace('Fully Paid', int(1),inplace=True)
df.loan_status.replace('Current', int(1),inplace=True)
df.loan_status.replace('Late (16-30 days)', int(0),inplace=True)
df.loan_status.replace('Late (31-120 days)', int(0),inplace=True)
df.loan_status.replace('Charged Off', np.nan,inplace=True)
df.loan_status.replace('In Grace Period', np.nan,inplace=True)
df.loan_status.replace('Default', np.nan,inplace=True)
```


```python
df.loan_status.value_counts()
```




    1.0    84496
    0.0     2274
    Name: loan_status, dtype: int64




```python
df.dropna(subset=['loan_status'],inplace=True)
```


```python
df.loan_status.value_counts()
```




    1.0    84496
    0.0     2274
    Name: loan_status, dtype: int64




```python
df.loan_status.isnull()
```




    0        False
    1        False
    2        False
    3        False
    5        False
    6        False
    7        False
    8        False
    10       False
    11       False
    12       False
    13       False
    14       False
    15       False
    16       False
    17       False
    18       False
    19       False
    20       False
    21       False
    22       False
    23       False
    24       False
    25       False
    26       False
    27       False
    28       False
    29       False
    31       False
    32       False
             ...  
    99087    False
    99088    False
    99089    False
    99090    False
    99091    False
    99092    False
    99093    False
    99094    False
    99095    False
    99097    False
    99098    False
    99099    False
    99100    False
    99101    False
    99102    False
    99103    False
    99104    False
    99105    False
    99106    False
    99107    False
    99108    False
    99109    False
    99111    False
    99112    False
    99113    False
    99115    False
    99116    False
    99117    False
    99118    False
    99119    False
    Name: loan_status, Length: 86770, dtype: bool




```python
df.revol_util = pd.Series(df.revol_util).str.replace('%', '').astype(float)
```

###  处理缺失值过多的类别型变量


```python
for col in df.select_dtypes(include=['object']).columns:
    print("Column {} has {} unique instances".format( col, len(df[col].unique())))
```

    Column term has 2 unique instances
    Column grade has 7 unique instances
    Column sub_grade has 35 unique instances
    Column home_ownership has 4 unique instances
    Column verification_status has 3 unique instances
    Column issue_d has 3 unique instances
    Column pymnt_plan has 2 unique instances
    Column desc has 6 unique instances
    Column purpose has 13 unique instances
    Column title has 13 unique instances
    Column earliest_cr_line has 608 unique instances
    Column initial_list_status has 2 unique instances
    Column last_pymnt_d has 25 unique instances
    Column next_pymnt_d has 3 unique instances
    Column last_credit_pull_d has 27 unique instances
    Column application_type has 2 unique instances
    Column verification_status_joint has 2 unique instances
    Column hardship_flag has 2 unique instances
    Column hardship_type has 2 unique instances
    Column hardship_reason has 10 unique instances
    Column hardship_status has 4 unique instances
    Column hardship_start_date has 16 unique instances
    Column hardship_end_date has 17 unique instances
    Column payment_plan_start_date has 16 unique instances
    Column hardship_loan_status has 5 unique instances
    Column disbursement_method has 2 unique instances
    Column debt_settlement_flag has 2 unique instances
    Column debt_settlement_flag_date has 15 unique instances
    Column settlement_status has 2 unique instances
    Column settlement_date has 16 unique instances
    


```python
# 判断缺失数据较多的列，并进行删除操作
df.select_dtypes(include=['object']).describe().T.assign(missing_pct=df.apply(lambda x : (len(x)-x.count())/float(len(x))))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
      <th>missing_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>term</th>
      <td>86770</td>
      <td>2</td>
      <td>36 months</td>
      <td>65451</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>grade</th>
      <td>86770</td>
      <td>7</td>
      <td>B</td>
      <td>29812</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>sub_grade</th>
      <td>86770</td>
      <td>35</td>
      <td>B5</td>
      <td>7472</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>home_ownership</th>
      <td>86770</td>
      <td>4</td>
      <td>MORTGAGE</td>
      <td>41716</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>verification_status</th>
      <td>86770</td>
      <td>3</td>
      <td>Source Verified</td>
      <td>37066</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>issue_d</th>
      <td>86770</td>
      <td>3</td>
      <td>Aug-2016</td>
      <td>31728</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>pymnt_plan</th>
      <td>86770</td>
      <td>2</td>
      <td>n</td>
      <td>86724</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>desc</th>
      <td>5</td>
      <td>5</td>
      <td>I have recently purchased and built a new home...</td>
      <td>1</td>
      <td>0.999942</td>
    </tr>
    <tr>
      <th>purpose</th>
      <td>86770</td>
      <td>13</td>
      <td>debt_consolidation</td>
      <td>50030</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>title</th>
      <td>82301</td>
      <td>12</td>
      <td>Debt consolidation</td>
      <td>47030</td>
      <td>0.051504</td>
    </tr>
    <tr>
      <th>earliest_cr_line</th>
      <td>86770</td>
      <td>608</td>
      <td>Aug-2004</td>
      <td>704</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>initial_list_status</th>
      <td>86770</td>
      <td>2</td>
      <td>w</td>
      <td>62962</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>last_pymnt_d</th>
      <td>86770</td>
      <td>25</td>
      <td>Jun-2018</td>
      <td>35077</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>next_pymnt_d</th>
      <td>54335</td>
      <td>2</td>
      <td>Jul-2018</td>
      <td>38520</td>
      <td>0.373804</td>
    </tr>
    <tr>
      <th>last_credit_pull_d</th>
      <td>86766</td>
      <td>26</td>
      <td>Jun-2018</td>
      <td>63805</td>
      <td>0.000046</td>
    </tr>
    <tr>
      <th>application_type</th>
      <td>86770</td>
      <td>2</td>
      <td>Individual</td>
      <td>86310</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>verification_status_joint</th>
      <td>460</td>
      <td>1</td>
      <td>Not Verified</td>
      <td>460</td>
      <td>0.994699</td>
    </tr>
    <tr>
      <th>hardship_flag</th>
      <td>86770</td>
      <td>2</td>
      <td>N</td>
      <td>86712</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>hardship_type</th>
      <td>589</td>
      <td>1</td>
      <td>INTEREST ONLY-3 MONTHS DEFERRAL</td>
      <td>589</td>
      <td>0.993212</td>
    </tr>
    <tr>
      <th>hardship_reason</th>
      <td>589</td>
      <td>9</td>
      <td>NATURAL_DISASTER</td>
      <td>263</td>
      <td>0.993212</td>
    </tr>
    <tr>
      <th>hardship_status</th>
      <td>589</td>
      <td>3</td>
      <td>COMPLETED</td>
      <td>493</td>
      <td>0.993212</td>
    </tr>
    <tr>
      <th>hardship_start_date</th>
      <td>589</td>
      <td>15</td>
      <td>Sep-2017</td>
      <td>214</td>
      <td>0.993212</td>
    </tr>
    <tr>
      <th>hardship_end_date</th>
      <td>589</td>
      <td>16</td>
      <td>Dec-2017</td>
      <td>154</td>
      <td>0.993212</td>
    </tr>
    <tr>
      <th>payment_plan_start_date</th>
      <td>589</td>
      <td>15</td>
      <td>Sep-2017</td>
      <td>152</td>
      <td>0.993212</td>
    </tr>
    <tr>
      <th>hardship_loan_status</th>
      <td>589</td>
      <td>4</td>
      <td>Late (16-30 days)</td>
      <td>234</td>
      <td>0.993212</td>
    </tr>
    <tr>
      <th>disbursement_method</th>
      <td>86770</td>
      <td>2</td>
      <td>Cash</td>
      <td>86745</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>debt_settlement_flag</th>
      <td>86770</td>
      <td>2</td>
      <td>N</td>
      <td>86683</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>debt_settlement_flag_date</th>
      <td>87</td>
      <td>14</td>
      <td>Jun-2018</td>
      <td>29</td>
      <td>0.998997</td>
    </tr>
    <tr>
      <th>settlement_status</th>
      <td>87</td>
      <td>1</td>
      <td>ACTIVE</td>
      <td>87</td>
      <td>0.998997</td>
    </tr>
    <tr>
      <th>settlement_date</th>
      <td>87</td>
      <td>15</td>
      <td>Jun-2018</td>
      <td>29</td>
      <td>0.998997</td>
    </tr>
  </tbody>
</table>
</div>



根据missing_pct和count这两列可以看出很多缺失的列，需要对这些列进行删除操作


```python
df.drop('desc',1,inplace=True)
df.drop('verification_status_joint',1,inplace=True)
df.drop(['hardship_type','hardship_reason','hardship_status','hardship_start_date','hardship_end_date','payment_plan_start_date','hardship_loan_status','debt_settlement_flag_date','settlement_date'],1,inplace=True)
```


```python
df.drop('earliest_cr_line',1,inplace=True)
df.drop('revol_util',1,inplace=True)
df.drop('purpose',1,inplace=True)
df.drop('title',1,inplace=True)
df.drop('term',1,inplace=True)
df.drop('issue_d',1,inplace=True)
# df.drop('',1,inplace=True)
# 贷后相关的字段
df.drop(['out_prncp','out_prncp_inv','total_pymnt',
         'total_pymnt_inv','total_rec_prncp', 'grade', 'sub_grade'] ,1, inplace=True)
df.drop(['total_rec_int','total_rec_late_fee',
         'recoveries','collection_recovery_fee',
         'collection_recovery_fee' ],1, inplace=True)
df.drop(['last_pymnt_d','last_pymnt_amnt',
         'next_pymnt_d','last_credit_pull_d'],1, inplace=True)
df.drop(['policy_code'],1, inplace=True)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 86770 entries, 0 to 99119
    Columns: 107 entries, loan_amnt to settlement_term
    dtypes: float64(97), int32(1), object(9)
    memory usage: 71.2+ MB
    

### 将数值型的缺失值较多的列进行删除


```python
df.select_dtypes(include=['float']).describe().T.\
assign(missing_pct=df.apply(lambda x : (len(x)-x.count())/float(len(x))))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>missing_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>loan_amnt</th>
      <td>86770.0</td>
      <td>13992.648381</td>
      <td>8860.658420</td>
      <td>1000.00</td>
      <td>7000.0000</td>
      <td>12000.00</td>
      <td>20000.0000</td>
      <td>40000.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>funded_amnt</th>
      <td>86770.0</td>
      <td>13992.648381</td>
      <td>8860.658420</td>
      <td>1000.00</td>
      <td>7000.0000</td>
      <td>12000.00</td>
      <td>20000.0000</td>
      <td>40000.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>funded_amnt_inv</th>
      <td>86770.0</td>
      <td>13988.150570</td>
      <td>8857.893095</td>
      <td>1000.00</td>
      <td>7000.0000</td>
      <td>12000.00</td>
      <td>20000.0000</td>
      <td>40000.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>int_rate</th>
      <td>86770.0</td>
      <td>13.339744</td>
      <td>4.662115</td>
      <td>5.32</td>
      <td>10.4900</td>
      <td>12.79</td>
      <td>15.5900</td>
      <td>30.99</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>installment</th>
      <td>86770.0</td>
      <td>425.346229</td>
      <td>269.732279</td>
      <td>30.12</td>
      <td>230.8000</td>
      <td>352.32</td>
      <td>558.3200</td>
      <td>1535.71</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>annual_inc</th>
      <td>86770.0</td>
      <td>79304.560449</td>
      <td>74486.852068</td>
      <td>0.00</td>
      <td>48000.0000</td>
      <td>67000.00</td>
      <td>95000.0000</td>
      <td>8400000.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>loan_status</th>
      <td>86770.0</td>
      <td>0.973793</td>
      <td>0.159752</td>
      <td>0.00</td>
      <td>1.0000</td>
      <td>1.00</td>
      <td>1.0000</td>
      <td>1.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>url</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>dti</th>
      <td>86767.0</td>
      <td>17.732272</td>
      <td>8.891494</td>
      <td>0.00</td>
      <td>11.7200</td>
      <td>17.35</td>
      <td>23.5900</td>
      <td>999.00</td>
      <td>0.000035</td>
    </tr>
    <tr>
      <th>delinq_2yrs</th>
      <td>86770.0</td>
      <td>0.378541</td>
      <td>0.982088</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>21.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>inq_last_6mths</th>
      <td>86770.0</td>
      <td>0.548047</td>
      <td>0.844625</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>1.0000</td>
      <td>5.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>mths_since_last_delinq</th>
      <td>46645.0</td>
      <td>33.288112</td>
      <td>21.837896</td>
      <td>0.00</td>
      <td>15.0000</td>
      <td>30.00</td>
      <td>48.0000</td>
      <td>142.00</td>
      <td>0.462429</td>
    </tr>
    <tr>
      <th>mths_since_last_record</th>
      <td>16967.0</td>
      <td>67.008428</td>
      <td>24.329262</td>
      <td>0.00</td>
      <td>51.0000</td>
      <td>70.00</td>
      <td>84.0000</td>
      <td>119.00</td>
      <td>0.804460</td>
    </tr>
    <tr>
      <th>open_acc</th>
      <td>86770.0</td>
      <td>11.677089</td>
      <td>5.730229</td>
      <td>1.00</td>
      <td>8.0000</td>
      <td>11.00</td>
      <td>14.0000</td>
      <td>86.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>pub_rec</th>
      <td>86770.0</td>
      <td>0.260620</td>
      <td>0.711654</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>61.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>revol_bal</th>
      <td>86770.0</td>
      <td>15700.789201</td>
      <td>21787.950268</td>
      <td>0.00</td>
      <td>5672.2500</td>
      <td>10539.50</td>
      <td>18653.0000</td>
      <td>876178.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>total_acc</th>
      <td>86770.0</td>
      <td>24.009623</td>
      <td>11.914658</td>
      <td>2.00</td>
      <td>15.0000</td>
      <td>22.00</td>
      <td>31.0000</td>
      <td>119.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>collections_12_mths_ex_med</th>
      <td>86770.0</td>
      <td>0.020871</td>
      <td>0.166700</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>10.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>mths_since_last_major_derog</th>
      <td>25416.0</td>
      <td>43.956996</td>
      <td>21.772484</td>
      <td>0.00</td>
      <td>27.0000</td>
      <td>43.00</td>
      <td>62.0000</td>
      <td>165.00</td>
      <td>0.707088</td>
    </tr>
    <tr>
      <th>annual_inc_joint</th>
      <td>460.0</td>
      <td>119731.355543</td>
      <td>51847.018868</td>
      <td>26943.12</td>
      <td>85000.0000</td>
      <td>111000.00</td>
      <td>145000.0000</td>
      <td>400000.00</td>
      <td>0.994699</td>
    </tr>
    <tr>
      <th>dti_joint</th>
      <td>460.0</td>
      <td>18.422152</td>
      <td>6.675737</td>
      <td>2.56</td>
      <td>13.9375</td>
      <td>18.26</td>
      <td>22.7875</td>
      <td>48.58</td>
      <td>0.994699</td>
    </tr>
    <tr>
      <th>acc_now_delinq</th>
      <td>86770.0</td>
      <td>0.006546</td>
      <td>0.085499</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>4.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>tot_coll_amt</th>
      <td>86770.0</td>
      <td>280.983439</td>
      <td>1825.358766</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>172575.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>tot_cur_bal</th>
      <td>86770.0</td>
      <td>141618.435081</td>
      <td>159319.917606</td>
      <td>0.00</td>
      <td>28983.0000</td>
      <td>79803.50</td>
      <td>211297.7500</td>
      <td>3764968.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>open_acc_6m</th>
      <td>86770.0</td>
      <td>0.949084</td>
      <td>1.154786</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>1.00</td>
      <td>1.0000</td>
      <td>13.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>open_act_il</th>
      <td>86770.0</td>
      <td>2.816123</td>
      <td>3.114175</td>
      <td>0.00</td>
      <td>1.0000</td>
      <td>2.00</td>
      <td>3.0000</td>
      <td>43.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>open_il_12m</th>
      <td>86770.0</td>
      <td>0.695678</td>
      <td>0.952599</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>1.0000</td>
      <td>11.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>open_il_24m</th>
      <td>86770.0</td>
      <td>1.578092</td>
      <td>1.621432</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>1.00</td>
      <td>2.0000</td>
      <td>26.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>mths_since_rcnt_il</th>
      <td>84447.0</td>
      <td>21.811160</td>
      <td>26.764207</td>
      <td>0.00</td>
      <td>7.0000</td>
      <td>13.00</td>
      <td>25.0000</td>
      <td>503.00</td>
      <td>0.026772</td>
    </tr>
    <tr>
      <th>total_bal_il</th>
      <td>86770.0</td>
      <td>34920.427164</td>
      <td>42123.969735</td>
      <td>0.00</td>
      <td>9014.2500</td>
      <td>22997.00</td>
      <td>45513.5000</td>
      <td>1547285.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>num_tl_op_past_12m</th>
      <td>86770.0</td>
      <td>2.191898</td>
      <td>1.931044</td>
      <td>0.00</td>
      <td>1.0000</td>
      <td>2.00</td>
      <td>3.0000</td>
      <td>24.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>pct_tl_nvr_dlq</th>
      <td>86770.0</td>
      <td>93.250410</td>
      <td>9.712188</td>
      <td>0.00</td>
      <td>90.0000</td>
      <td>96.95</td>
      <td>100.0000</td>
      <td>100.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>percent_bc_gt_75</th>
      <td>85791.0</td>
      <td>42.178564</td>
      <td>36.295768</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>33.30</td>
      <td>70.0000</td>
      <td>100.00</td>
      <td>0.011283</td>
    </tr>
    <tr>
      <th>pub_rec_bankruptcies</th>
      <td>86770.0</td>
      <td>0.146318</td>
      <td>0.403204</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>8.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>tax_liens</th>
      <td>86770.0</td>
      <td>0.074104</td>
      <td>0.511770</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>61.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>tot_hi_cred_lim</th>
      <td>86770.0</td>
      <td>175777.763858</td>
      <td>178392.180843</td>
      <td>2500.00</td>
      <td>49905.7500</td>
      <td>111976.00</td>
      <td>253691.5000</td>
      <td>3953111.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>total_bal_ex_mort</th>
      <td>86770.0</td>
      <td>50888.933214</td>
      <td>49440.559933</td>
      <td>0.00</td>
      <td>20803.0000</td>
      <td>37731.00</td>
      <td>64243.2500</td>
      <td>1548128.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>total_bc_limit</th>
      <td>86770.0</td>
      <td>21251.124029</td>
      <td>21050.690545</td>
      <td>0.00</td>
      <td>7800.0000</td>
      <td>15000.00</td>
      <td>27500.0000</td>
      <td>520500.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>total_il_high_credit_limit</th>
      <td>86770.0</td>
      <td>44113.311882</td>
      <td>44747.921766</td>
      <td>0.00</td>
      <td>15709.2500</td>
      <td>33192.50</td>
      <td>59000.0000</td>
      <td>2000000.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>revol_bal_joint</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>sec_app_earliest_cr_line</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>sec_app_inq_last_6mths</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>sec_app_mort_acc</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>sec_app_open_acc</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>sec_app_revol_util</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>sec_app_open_act_il</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>sec_app_num_rev_accts</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>sec_app_chargeoff_within_12_mths</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>sec_app_collections_12_mths_ex_med</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>sec_app_mths_since_last_major_derog</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>deferral_term</th>
      <td>589.0</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>3.00</td>
      <td>3.0000</td>
      <td>3.00</td>
      <td>3.0000</td>
      <td>3.00</td>
      <td>0.993212</td>
    </tr>
    <tr>
      <th>hardship_amount</th>
      <td>589.0</td>
      <td>163.295331</td>
      <td>131.454229</td>
      <td>6.35</td>
      <td>67.0700</td>
      <td>125.10</td>
      <td>219.1100</td>
      <td>769.03</td>
      <td>0.993212</td>
    </tr>
    <tr>
      <th>hardship_length</th>
      <td>589.0</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>3.00</td>
      <td>3.0000</td>
      <td>3.00</td>
      <td>3.0000</td>
      <td>3.00</td>
      <td>0.993212</td>
    </tr>
    <tr>
      <th>hardship_dpd</th>
      <td>589.0</td>
      <td>11.607810</td>
      <td>10.522442</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>12.00</td>
      <td>22.0000</td>
      <td>37.00</td>
      <td>0.993212</td>
    </tr>
    <tr>
      <th>orig_projected_additional_accrued_interest</th>
      <td>551.0</td>
      <td>495.658911</td>
      <td>393.610753</td>
      <td>19.05</td>
      <td>208.3950</td>
      <td>387.63</td>
      <td>666.8700</td>
      <td>2307.09</td>
      <td>0.993650</td>
    </tr>
    <tr>
      <th>hardship_payoff_balance_amount</th>
      <td>589.0</td>
      <td>12088.823175</td>
      <td>7282.730741</td>
      <td>750.93</td>
      <td>6528.8700</td>
      <td>10569.61</td>
      <td>16276.7300</td>
      <td>33494.15</td>
      <td>0.993212</td>
    </tr>
    <tr>
      <th>hardship_last_payment_amount</th>
      <td>589.0</td>
      <td>192.151002</td>
      <td>207.123799</td>
      <td>0.02</td>
      <td>35.2400</td>
      <td>121.84</td>
      <td>285.4700</td>
      <td>1223.33</td>
      <td>0.993212</td>
    </tr>
    <tr>
      <th>settlement_amount</th>
      <td>87.0</td>
      <td>5355.673333</td>
      <td>4012.832699</td>
      <td>508.00</td>
      <td>2082.0000</td>
      <td>4351.00</td>
      <td>7535.5000</td>
      <td>17191.00</td>
      <td>0.998997</td>
    </tr>
    <tr>
      <th>settlement_percentage</th>
      <td>87.0</td>
      <td>53.376092</td>
      <td>4.694325</td>
      <td>49.99</td>
      <td>50.0000</td>
      <td>50.01</td>
      <td>55.0000</td>
      <td>65.00</td>
      <td>0.998997</td>
    </tr>
    <tr>
      <th>settlement_term</th>
      <td>87.0</td>
      <td>13.747126</td>
      <td>4.394019</td>
      <td>1.00</td>
      <td>10.0000</td>
      <td>15.00</td>
      <td>18.0000</td>
      <td>18.00</td>
      <td>0.998997</td>
    </tr>
  </tbody>
</table>
<p>97 rows × 9 columns</p>
</div>




```python
df.drop('annual_inc_joint',1,inplace=True)
df.drop('dti_joint',1,inplace=True)
df.drop('url',1,inplace=True)
df.drop(['revol_bal_joint','sec_app_earliest_cr_line','sec_app_inq_last_6mths','sec_app_mort_acc','sec_app_open_acc','sec_app_revol_util','sec_app_open_act_il','sec_app_num_rev_accts','sec_app_chargeoff_within_12_mths','sec_app_collections_12_mths_ex_med','sec_app_mths_since_last_major_derog'],1,inplace=True)
```


```python
df.select_dtypes(include=['float']).describe().T.\
assign(missing_pct=df.apply(lambda x : (len(x)-x.count())/float(len(x))))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>missing_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>loan_amnt</th>
      <td>86770.0</td>
      <td>13992.648381</td>
      <td>8860.658420</td>
      <td>1000.00</td>
      <td>7000.000</td>
      <td>12000.00</td>
      <td>20000.00</td>
      <td>40000.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>funded_amnt</th>
      <td>86770.0</td>
      <td>13992.648381</td>
      <td>8860.658420</td>
      <td>1000.00</td>
      <td>7000.000</td>
      <td>12000.00</td>
      <td>20000.00</td>
      <td>40000.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>funded_amnt_inv</th>
      <td>86770.0</td>
      <td>13988.150570</td>
      <td>8857.893095</td>
      <td>1000.00</td>
      <td>7000.000</td>
      <td>12000.00</td>
      <td>20000.00</td>
      <td>40000.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>int_rate</th>
      <td>86770.0</td>
      <td>13.339744</td>
      <td>4.662115</td>
      <td>5.32</td>
      <td>10.490</td>
      <td>12.79</td>
      <td>15.59</td>
      <td>30.99</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>installment</th>
      <td>86770.0</td>
      <td>425.346229</td>
      <td>269.732279</td>
      <td>30.12</td>
      <td>230.800</td>
      <td>352.32</td>
      <td>558.32</td>
      <td>1535.71</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>annual_inc</th>
      <td>86770.0</td>
      <td>79304.560449</td>
      <td>74486.852068</td>
      <td>0.00</td>
      <td>48000.000</td>
      <td>67000.00</td>
      <td>95000.00</td>
      <td>8400000.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>loan_status</th>
      <td>86770.0</td>
      <td>0.973793</td>
      <td>0.159752</td>
      <td>0.00</td>
      <td>1.000</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>dti</th>
      <td>86767.0</td>
      <td>17.732272</td>
      <td>8.891494</td>
      <td>0.00</td>
      <td>11.720</td>
      <td>17.35</td>
      <td>23.59</td>
      <td>999.00</td>
      <td>0.000035</td>
    </tr>
    <tr>
      <th>delinq_2yrs</th>
      <td>86770.0</td>
      <td>0.378541</td>
      <td>0.982088</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>21.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>inq_last_6mths</th>
      <td>86770.0</td>
      <td>0.548047</td>
      <td>0.844625</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>5.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>mths_since_last_delinq</th>
      <td>46645.0</td>
      <td>33.288112</td>
      <td>21.837896</td>
      <td>0.00</td>
      <td>15.000</td>
      <td>30.00</td>
      <td>48.00</td>
      <td>142.00</td>
      <td>0.462429</td>
    </tr>
    <tr>
      <th>mths_since_last_record</th>
      <td>16967.0</td>
      <td>67.008428</td>
      <td>24.329262</td>
      <td>0.00</td>
      <td>51.000</td>
      <td>70.00</td>
      <td>84.00</td>
      <td>119.00</td>
      <td>0.804460</td>
    </tr>
    <tr>
      <th>open_acc</th>
      <td>86770.0</td>
      <td>11.677089</td>
      <td>5.730229</td>
      <td>1.00</td>
      <td>8.000</td>
      <td>11.00</td>
      <td>14.00</td>
      <td>86.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>pub_rec</th>
      <td>86770.0</td>
      <td>0.260620</td>
      <td>0.711654</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>61.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>revol_bal</th>
      <td>86770.0</td>
      <td>15700.789201</td>
      <td>21787.950268</td>
      <td>0.00</td>
      <td>5672.250</td>
      <td>10539.50</td>
      <td>18653.00</td>
      <td>876178.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>total_acc</th>
      <td>86770.0</td>
      <td>24.009623</td>
      <td>11.914658</td>
      <td>2.00</td>
      <td>15.000</td>
      <td>22.00</td>
      <td>31.00</td>
      <td>119.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>collections_12_mths_ex_med</th>
      <td>86770.0</td>
      <td>0.020871</td>
      <td>0.166700</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>10.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>mths_since_last_major_derog</th>
      <td>25416.0</td>
      <td>43.956996</td>
      <td>21.772484</td>
      <td>0.00</td>
      <td>27.000</td>
      <td>43.00</td>
      <td>62.00</td>
      <td>165.00</td>
      <td>0.707088</td>
    </tr>
    <tr>
      <th>acc_now_delinq</th>
      <td>86770.0</td>
      <td>0.006546</td>
      <td>0.085499</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>tot_coll_amt</th>
      <td>86770.0</td>
      <td>280.983439</td>
      <td>1825.358766</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>172575.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>tot_cur_bal</th>
      <td>86770.0</td>
      <td>141618.435081</td>
      <td>159319.917606</td>
      <td>0.00</td>
      <td>28983.000</td>
      <td>79803.50</td>
      <td>211297.75</td>
      <td>3764968.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>open_acc_6m</th>
      <td>86770.0</td>
      <td>0.949084</td>
      <td>1.154786</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>13.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>open_act_il</th>
      <td>86770.0</td>
      <td>2.816123</td>
      <td>3.114175</td>
      <td>0.00</td>
      <td>1.000</td>
      <td>2.00</td>
      <td>3.00</td>
      <td>43.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>open_il_12m</th>
      <td>86770.0</td>
      <td>0.695678</td>
      <td>0.952599</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>11.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>open_il_24m</th>
      <td>86770.0</td>
      <td>1.578092</td>
      <td>1.621432</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>1.00</td>
      <td>2.00</td>
      <td>26.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>mths_since_rcnt_il</th>
      <td>84447.0</td>
      <td>21.811160</td>
      <td>26.764207</td>
      <td>0.00</td>
      <td>7.000</td>
      <td>13.00</td>
      <td>25.00</td>
      <td>503.00</td>
      <td>0.026772</td>
    </tr>
    <tr>
      <th>total_bal_il</th>
      <td>86770.0</td>
      <td>34920.427164</td>
      <td>42123.969735</td>
      <td>0.00</td>
      <td>9014.250</td>
      <td>22997.00</td>
      <td>45513.50</td>
      <td>1547285.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>il_util</th>
      <td>74592.0</td>
      <td>71.086001</td>
      <td>23.439148</td>
      <td>0.00</td>
      <td>58.000</td>
      <td>74.00</td>
      <td>87.00</td>
      <td>1000.00</td>
      <td>0.140348</td>
    </tr>
    <tr>
      <th>open_rv_12m</th>
      <td>86770.0</td>
      <td>1.372882</td>
      <td>1.546456</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>1.00</td>
      <td>2.00</td>
      <td>20.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>open_rv_24m</th>
      <td>86770.0</td>
      <td>2.884430</td>
      <td>2.635246</td>
      <td>0.00</td>
      <td>1.000</td>
      <td>2.00</td>
      <td>4.00</td>
      <td>60.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>num_actv_rev_tl</th>
      <td>86770.0</td>
      <td>5.580143</td>
      <td>3.375133</td>
      <td>0.00</td>
      <td>3.000</td>
      <td>5.00</td>
      <td>7.00</td>
      <td>59.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>num_bc_sats</th>
      <td>86770.0</td>
      <td>4.626288</td>
      <td>2.999892</td>
      <td>0.00</td>
      <td>3.000</td>
      <td>4.00</td>
      <td>6.00</td>
      <td>61.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>num_bc_tl</th>
      <td>86770.0</td>
      <td>7.407053</td>
      <td>4.529108</td>
      <td>0.00</td>
      <td>4.000</td>
      <td>7.00</td>
      <td>10.00</td>
      <td>67.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>num_il_tl</th>
      <td>86770.0</td>
      <td>8.568918</td>
      <td>7.510092</td>
      <td>0.00</td>
      <td>3.000</td>
      <td>7.00</td>
      <td>11.00</td>
      <td>107.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>num_op_rev_tl</th>
      <td>86770.0</td>
      <td>8.155215</td>
      <td>4.699856</td>
      <td>0.00</td>
      <td>5.000</td>
      <td>7.00</td>
      <td>10.00</td>
      <td>79.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>num_rev_accts</th>
      <td>86770.0</td>
      <td>13.695909</td>
      <td>7.937062</td>
      <td>2.00</td>
      <td>8.000</td>
      <td>12.00</td>
      <td>18.00</td>
      <td>104.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>num_rev_tl_bal_gt_0</th>
      <td>86770.0</td>
      <td>5.524144</td>
      <td>3.262540</td>
      <td>0.00</td>
      <td>3.000</td>
      <td>5.00</td>
      <td>7.00</td>
      <td>59.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>num_sats</th>
      <td>86770.0</td>
      <td>11.633537</td>
      <td>5.709296</td>
      <td>1.00</td>
      <td>8.000</td>
      <td>11.00</td>
      <td>14.00</td>
      <td>85.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>num_tl_120dpd_2m</th>
      <td>83916.0</td>
      <td>0.001096</td>
      <td>0.035858</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4.00</td>
      <td>0.032892</td>
    </tr>
    <tr>
      <th>num_tl_30dpd</th>
      <td>86770.0</td>
      <td>0.004218</td>
      <td>0.067253</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>num_tl_90g_dpd_24m</th>
      <td>86770.0</td>
      <td>0.100000</td>
      <td>0.563002</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>20.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>num_tl_op_past_12m</th>
      <td>86770.0</td>
      <td>2.191898</td>
      <td>1.931044</td>
      <td>0.00</td>
      <td>1.000</td>
      <td>2.00</td>
      <td>3.00</td>
      <td>24.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>pct_tl_nvr_dlq</th>
      <td>86770.0</td>
      <td>93.250410</td>
      <td>9.712188</td>
      <td>0.00</td>
      <td>90.000</td>
      <td>96.95</td>
      <td>100.00</td>
      <td>100.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>percent_bc_gt_75</th>
      <td>85791.0</td>
      <td>42.178564</td>
      <td>36.295768</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>33.30</td>
      <td>70.00</td>
      <td>100.00</td>
      <td>0.011283</td>
    </tr>
    <tr>
      <th>pub_rec_bankruptcies</th>
      <td>86770.0</td>
      <td>0.146318</td>
      <td>0.403204</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>8.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>tax_liens</th>
      <td>86770.0</td>
      <td>0.074104</td>
      <td>0.511770</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>61.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>tot_hi_cred_lim</th>
      <td>86770.0</td>
      <td>175777.763858</td>
      <td>178392.180843</td>
      <td>2500.00</td>
      <td>49905.750</td>
      <td>111976.00</td>
      <td>253691.50</td>
      <td>3953111.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>total_bal_ex_mort</th>
      <td>86770.0</td>
      <td>50888.933214</td>
      <td>49440.559933</td>
      <td>0.00</td>
      <td>20803.000</td>
      <td>37731.00</td>
      <td>64243.25</td>
      <td>1548128.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>total_bc_limit</th>
      <td>86770.0</td>
      <td>21251.124029</td>
      <td>21050.690545</td>
      <td>0.00</td>
      <td>7800.000</td>
      <td>15000.00</td>
      <td>27500.00</td>
      <td>520500.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>total_il_high_credit_limit</th>
      <td>86770.0</td>
      <td>44113.311882</td>
      <td>44747.921766</td>
      <td>0.00</td>
      <td>15709.250</td>
      <td>33192.50</td>
      <td>59000.00</td>
      <td>2000000.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>deferral_term</th>
      <td>589.0</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>3.00</td>
      <td>3.000</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>0.993212</td>
    </tr>
    <tr>
      <th>hardship_amount</th>
      <td>589.0</td>
      <td>163.295331</td>
      <td>131.454229</td>
      <td>6.35</td>
      <td>67.070</td>
      <td>125.10</td>
      <td>219.11</td>
      <td>769.03</td>
      <td>0.993212</td>
    </tr>
    <tr>
      <th>hardship_length</th>
      <td>589.0</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>3.00</td>
      <td>3.000</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>0.993212</td>
    </tr>
    <tr>
      <th>hardship_dpd</th>
      <td>589.0</td>
      <td>11.607810</td>
      <td>10.522442</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>12.00</td>
      <td>22.00</td>
      <td>37.00</td>
      <td>0.993212</td>
    </tr>
    <tr>
      <th>orig_projected_additional_accrued_interest</th>
      <td>551.0</td>
      <td>495.658911</td>
      <td>393.610753</td>
      <td>19.05</td>
      <td>208.395</td>
      <td>387.63</td>
      <td>666.87</td>
      <td>2307.09</td>
      <td>0.993650</td>
    </tr>
    <tr>
      <th>hardship_payoff_balance_amount</th>
      <td>589.0</td>
      <td>12088.823175</td>
      <td>7282.730741</td>
      <td>750.93</td>
      <td>6528.870</td>
      <td>10569.61</td>
      <td>16276.73</td>
      <td>33494.15</td>
      <td>0.993212</td>
    </tr>
    <tr>
      <th>hardship_last_payment_amount</th>
      <td>589.0</td>
      <td>192.151002</td>
      <td>207.123799</td>
      <td>0.02</td>
      <td>35.240</td>
      <td>121.84</td>
      <td>285.47</td>
      <td>1223.33</td>
      <td>0.993212</td>
    </tr>
    <tr>
      <th>settlement_amount</th>
      <td>87.0</td>
      <td>5355.673333</td>
      <td>4012.832699</td>
      <td>508.00</td>
      <td>2082.000</td>
      <td>4351.00</td>
      <td>7535.50</td>
      <td>17191.00</td>
      <td>0.998997</td>
    </tr>
    <tr>
      <th>settlement_percentage</th>
      <td>87.0</td>
      <td>53.376092</td>
      <td>4.694325</td>
      <td>49.99</td>
      <td>50.000</td>
      <td>50.01</td>
      <td>55.00</td>
      <td>65.00</td>
      <td>0.998997</td>
    </tr>
    <tr>
      <th>settlement_term</th>
      <td>87.0</td>
      <td>13.747126</td>
      <td>4.394019</td>
      <td>1.00</td>
      <td>10.000</td>
      <td>15.00</td>
      <td>18.00</td>
      <td>18.00</td>
      <td>0.998997</td>
    </tr>
  </tbody>
</table>
<p>83 rows × 9 columns</p>
</div>




```python
df.select_dtypes(include=['float']).describe().T[df.select_dtypes(include=['float']).describe().T.\
assign(missing_pct=df.apply(lambda x : (len(x)-x.count())/float(len(x)))).isnull()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>loan_amnt</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>int_rate</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>annual_inc</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>loan_status</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>dti</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>delinq_2yrs</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>inq_last_6mths</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mths_since_last_delinq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mths_since_last_record</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>open_acc</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>pub_rec</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>revol_bal</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>total_acc</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>collections_12_mths_ex_med</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mths_since_last_major_derog</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>acc_now_delinq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>tot_coll_amt</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>tot_cur_bal</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>open_acc_6m</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>open_act_il</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>open_il_12m</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>open_il_24m</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mths_since_rcnt_il</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>total_bal_il</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>il_util</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>open_rv_12m</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>open_rv_24m</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max_bal_bc</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>all_util</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>total_rev_hi_lim</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>num_actv_rev_tl</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>num_bc_sats</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>num_bc_tl</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>num_il_tl</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>num_op_rev_tl</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>num_rev_accts</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>num_rev_tl_bal_gt_0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>num_sats</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>num_tl_120dpd_2m</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>num_tl_30dpd</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>num_tl_90g_dpd_24m</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>num_tl_op_past_12m</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>pct_tl_nvr_dlq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>percent_bc_gt_75</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>pub_rec_bankruptcies</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>tax_liens</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>tot_hi_cred_lim</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>total_bal_ex_mort</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>total_bc_limit</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>total_il_high_credit_limit</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>deferral_term</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>hardship_amount</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>hardship_length</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>hardship_dpd</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>orig_projected_additional_accrued_interest</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>hardship_payoff_balance_amount</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>hardship_last_payment_amount</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>settlement_amount</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>settlement_percentage</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>settlement_term</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>80 rows × 8 columns</p>
</div>



### 处理整数变量的列


```python
df.select_dtypes(include=['int']).describe().T.\
assign(missing_pct=df.apply(lambda x : (len(x)-x.count())/float(len(x))))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>missing_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>emp_length</th>
      <td>86770.0</td>
      <td>5.793604</td>
      <td>3.761932</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



### 处理高度线性相关的列


```python
cor = df.corr()
```


```python
cor.loc[:,:] = np.tril(cor,k=-1)
```


```python
cor
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>emp_length</th>
      <th>annual_inc</th>
      <th>loan_status</th>
      <th>dti</th>
      <th>delinq_2yrs</th>
      <th>...</th>
      <th>deferral_term</th>
      <th>hardship_amount</th>
      <th>hardship_length</th>
      <th>hardship_dpd</th>
      <th>orig_projected_additional_accrued_interest</th>
      <th>hardship_payoff_balance_amount</th>
      <th>hardship_last_payment_amount</th>
      <th>settlement_amount</th>
      <th>settlement_percentage</th>
      <th>settlement_term</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>loan_amnt</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>funded_amnt</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>funded_amnt_inv</th>
      <td>0.999993</td>
      <td>0.999993</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>int_rate</th>
      <td>0.183818</td>
      <td>0.183818</td>
      <td>0.183990</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>installment</th>
      <td>0.953658</td>
      <td>0.953658</td>
      <td>0.953570</td>
      <td>0.207638</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>emp_length</th>
      <td>0.110753</td>
      <td>0.110753</td>
      <td>0.110799</td>
      <td>-0.018173</td>
      <td>0.096081</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>annual_inc</th>
      <td>0.314069</td>
      <td>0.314069</td>
      <td>0.314033</td>
      <td>-0.066962</td>
      <td>0.295867</td>
      <td>0.094171</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>loan_status</th>
      <td>-0.013152</td>
      <td>-0.013152</td>
      <td>-0.013179</td>
      <td>-0.080925</td>
      <td>-0.018661</td>
      <td>0.013513</td>
      <td>0.016563</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>dti</th>
      <td>0.038334</td>
      <td>0.038334</td>
      <td>0.038344</td>
      <td>0.167446</td>
      <td>0.048354</td>
      <td>0.011855</td>
      <td>-0.145502</td>
      <td>-0.022552</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>delinq_2yrs</th>
      <td>-0.008369</td>
      <td>-0.008369</td>
      <td>-0.008387</td>
      <td>0.029667</td>
      <td>-0.002644</td>
      <td>0.026023</td>
      <td>0.037214</td>
      <td>-0.020069</td>
      <td>-0.012498</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>inq_last_6mths</th>
      <td>-0.016169</td>
      <td>-0.016169</td>
      <td>-0.016198</td>
      <td>0.170607</td>
      <td>0.007442</td>
      <td>0.001574</td>
      <td>0.034509</td>
      <td>-0.019708</td>
      <td>-0.001975</td>
      <td>0.025702</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>mths_since_last_delinq</th>
      <td>0.002074</td>
      <td>0.002074</td>
      <td>0.002101</td>
      <td>-0.018567</td>
      <td>-0.004217</td>
      <td>-0.016531</td>
      <td>-0.033823</td>
      <td>0.013422</td>
      <td>0.009132</td>
      <td>-0.551436</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>mths_since_last_record</th>
      <td>-0.015390</td>
      <td>-0.015390</td>
      <td>-0.015358</td>
      <td>-0.013327</td>
      <td>-0.023256</td>
      <td>0.027087</td>
      <td>-0.080178</td>
      <td>-0.000778</td>
      <td>0.049695</td>
      <td>-0.075914</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>open_acc</th>
      <td>0.189283</td>
      <td>0.189283</td>
      <td>0.189256</td>
      <td>-0.006746</td>
      <td>0.176980</td>
      <td>0.051260</td>
      <td>0.140723</td>
      <td>-0.006664</td>
      <td>0.261796</td>
      <td>0.056525</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>pub_rec</th>
      <td>-0.039875</td>
      <td>-0.039875</td>
      <td>-0.039901</td>
      <td>0.049172</td>
      <td>-0.029240</td>
      <td>0.012645</td>
      <td>0.000980</td>
      <td>-0.011186</td>
      <td>-0.051349</td>
      <td>-0.039383</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>revol_bal</th>
      <td>0.323861</td>
      <td>0.323861</td>
      <td>0.323821</td>
      <td>-0.023577</td>
      <td>0.306485</td>
      <td>0.088163</td>
      <td>0.308800</td>
      <td>0.018380</td>
      <td>0.131510</td>
      <td>-0.025537</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>total_acc</th>
      <td>0.207745</td>
      <td>0.207745</td>
      <td>0.207719</td>
      <td>-0.046995</td>
      <td>0.183350</td>
      <td>0.088303</td>
      <td>0.176989</td>
      <td>0.003432</td>
      <td>0.219191</td>
      <td>0.121100</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>collections_12_mths_ex_med</th>
      <td>-0.020581</td>
      <td>-0.020581</td>
      <td>-0.020595</td>
      <td>0.015218</td>
      <td>-0.019196</td>
      <td>-0.009854</td>
      <td>-0.010283</td>
      <td>-0.011052</td>
      <td>-0.006429</td>
      <td>0.082467</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>mths_since_last_major_derog</th>
      <td>0.029221</td>
      <td>0.029221</td>
      <td>0.029253</td>
      <td>-0.010362</td>
      <td>0.023953</td>
      <td>0.011823</td>
      <td>-0.006633</td>
      <td>0.008867</td>
      <td>0.022977</td>
      <td>-0.439524</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>acc_now_delinq</th>
      <td>-0.000643</td>
      <td>-0.000643</td>
      <td>-0.000652</td>
      <td>0.011462</td>
      <td>0.000688</td>
      <td>0.012119</td>
      <td>0.011594</td>
      <td>-0.000940</td>
      <td>-0.002557</td>
      <td>0.122841</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>tot_coll_amt</th>
      <td>-0.024839</td>
      <td>-0.024839</td>
      <td>-0.024834</td>
      <td>0.007813</td>
      <td>-0.020357</td>
      <td>0.002719</td>
      <td>-0.003110</td>
      <td>0.003976</td>
      <td>-0.016754</td>
      <td>-0.003642</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>tot_cur_bal</th>
      <td>0.306866</td>
      <td>0.306866</td>
      <td>0.306888</td>
      <td>-0.067493</td>
      <td>0.270260</td>
      <td>0.117314</td>
      <td>0.392770</td>
      <td>0.022304</td>
      <td>0.022050</td>
      <td>0.056121</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>open_acc_6m</th>
      <td>-0.017690</td>
      <td>-0.017690</td>
      <td>-0.017694</td>
      <td>0.155748</td>
      <td>0.003937</td>
      <td>0.017539</td>
      <td>0.046791</td>
      <td>-0.024538</td>
      <td>0.040895</td>
      <td>0.000350</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>open_act_il</th>
      <td>0.021932</td>
      <td>0.021932</td>
      <td>0.021922</td>
      <td>0.031291</td>
      <td>0.016411</td>
      <td>-0.087070</td>
      <td>0.072141</td>
      <td>-0.005725</td>
      <td>0.222380</td>
      <td>0.071720</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>open_il_12m</th>
      <td>0.009230</td>
      <td>0.009230</td>
      <td>0.009251</td>
      <td>0.185548</td>
      <td>0.023938</td>
      <td>0.042698</td>
      <td>0.093255</td>
      <td>-0.021131</td>
      <td>0.166589</td>
      <td>-0.005855</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>open_il_24m</th>
      <td>0.037502</td>
      <td>0.037502</td>
      <td>0.037524</td>
      <td>0.163407</td>
      <td>0.044889</td>
      <td>0.046481</td>
      <td>0.115768</td>
      <td>-0.021197</td>
      <td>0.229569</td>
      <td>-0.023145</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>mths_since_rcnt_il</th>
      <td>-0.026022</td>
      <td>-0.026022</td>
      <td>-0.026030</td>
      <td>-0.097319</td>
      <td>-0.027290</td>
      <td>-0.033338</td>
      <td>-0.076912</td>
      <td>0.015779</td>
      <td>-0.244153</td>
      <td>0.011909</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>total_bal_il</th>
      <td>0.145235</td>
      <td>0.145235</td>
      <td>0.145235</td>
      <td>0.037161</td>
      <td>0.131976</td>
      <td>-0.023716</td>
      <td>0.224390</td>
      <td>-0.000659</td>
      <td>0.239179</td>
      <td>0.059929</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>il_util</th>
      <td>-0.088801</td>
      <td>-0.088801</td>
      <td>-0.088763</td>
      <td>0.146080</td>
      <td>-0.080543</td>
      <td>-0.088790</td>
      <td>-0.034212</td>
      <td>-0.027497</td>
      <td>-0.009064</td>
      <td>-0.006509</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>open_rv_12m</th>
      <td>-0.039301</td>
      <td>-0.039301</td>
      <td>-0.039299</td>
      <td>0.134080</td>
      <td>-0.010669</td>
      <td>0.004968</td>
      <td>-0.000835</td>
      <td>-0.028553</td>
      <td>0.010479</td>
      <td>-0.026754</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>num_actv_rev_tl</th>
      <td>0.175949</td>
      <td>0.175949</td>
      <td>0.175932</td>
      <td>0.056059</td>
      <td>0.176815</td>
      <td>0.109311</td>
      <td>0.088747</td>
      <td>-0.019766</td>
      <td>0.229290</td>
      <td>-0.003791</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>num_bc_sats</th>
      <td>0.236664</td>
      <td>0.236664</td>
      <td>0.236621</td>
      <td>-0.045445</td>
      <td>0.224395</td>
      <td>0.067978</td>
      <td>0.135832</td>
      <td>-0.001078</td>
      <td>0.110538</td>
      <td>-0.035852</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>num_bc_tl</th>
      <td>0.221586</td>
      <td>0.221586</td>
      <td>0.221538</td>
      <td>-0.085157</td>
      <td>0.206882</td>
      <td>0.092169</td>
      <td>0.145879</td>
      <td>0.007720</td>
      <td>0.083480</td>
      <td>0.029768</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>num_il_tl</th>
      <td>0.071276</td>
      <td>0.071276</td>
      <td>0.071275</td>
      <td>0.015775</td>
      <td>0.056311</td>
      <td>-0.021343</td>
      <td>0.103582</td>
      <td>-0.006861</td>
      <td>0.209211</td>
      <td>0.092056</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>num_op_rev_tl</th>
      <td>0.180233</td>
      <td>0.180233</td>
      <td>0.180201</td>
      <td>-0.016180</td>
      <td>0.174405</td>
      <td>0.097503</td>
      <td>0.085403</td>
      <td>-0.007875</td>
      <td>0.168069</td>
      <td>0.008976</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>num_rev_accts</th>
      <td>0.187665</td>
      <td>0.187665</td>
      <td>0.187622</td>
      <td>-0.064875</td>
      <td>0.173955</td>
      <td>0.116064</td>
      <td>0.110810</td>
      <td>0.005822</td>
      <td>0.135595</td>
      <td>0.075864</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>num_rev_tl_bal_gt_0</th>
      <td>0.173031</td>
      <td>0.173031</td>
      <td>0.173023</td>
      <td>0.057363</td>
      <td>0.175158</td>
      <td>0.109516</td>
      <td>0.087066</td>
      <td>-0.019129</td>
      <td>0.232890</td>
      <td>-0.006032</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>num_sats</th>
      <td>0.189243</td>
      <td>0.189243</td>
      <td>0.189216</td>
      <td>-0.006886</td>
      <td>0.177005</td>
      <td>0.050780</td>
      <td>0.140348</td>
      <td>-0.006524</td>
      <td>0.261779</td>
      <td>0.054582</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>num_tl_120dpd_2m</th>
      <td>-0.003400</td>
      <td>-0.003400</td>
      <td>-0.003416</td>
      <td>0.001534</td>
      <td>-0.003172</td>
      <td>0.003219</td>
      <td>0.003622</td>
      <td>0.002894</td>
      <td>-0.011306</td>
      <td>0.046662</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>num_tl_30dpd</th>
      <td>0.005174</td>
      <td>0.005174</td>
      <td>0.005163</td>
      <td>0.010414</td>
      <td>0.006622</td>
      <td>0.012415</td>
      <td>0.012151</td>
      <td>-0.001511</td>
      <td>0.002556</td>
      <td>0.102156</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>num_tl_90g_dpd_24m</th>
      <td>-0.023556</td>
      <td>-0.023556</td>
      <td>-0.023574</td>
      <td>0.021237</td>
      <td>-0.019500</td>
      <td>-0.002221</td>
      <td>0.006915</td>
      <td>-0.012634</td>
      <td>-0.015192</td>
      <td>0.670425</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>num_tl_op_past_12m</th>
      <td>-0.014006</td>
      <td>-0.014006</td>
      <td>-0.013991</td>
      <td>0.196019</td>
      <td>0.013791</td>
      <td>0.027795</td>
      <td>0.059326</td>
      <td>-0.032077</td>
      <td>0.080556</td>
      <td>-0.029705</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>pct_tl_nvr_dlq</th>
      <td>0.083337</td>
      <td>0.083337</td>
      <td>0.083360</td>
      <td>-0.041184</td>
      <td>0.068789</td>
      <td>-0.024941</td>
      <td>-0.008379</td>
      <td>0.022224</td>
      <td>0.086247</td>
      <td>-0.460886</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>percent_bc_gt_75</th>
      <td>0.035651</td>
      <td>0.035651</td>
      <td>0.035702</td>
      <td>0.196573</td>
      <td>0.051238</td>
      <td>0.032716</td>
      <td>0.003882</td>
      <td>-0.017767</td>
      <td>0.145310</td>
      <td>-0.004343</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>pub_rec_bankruptcies</th>
      <td>-0.070945</td>
      <td>-0.070945</td>
      <td>-0.070962</td>
      <td>0.051258</td>
      <td>-0.062869</td>
      <td>0.003886</td>
      <td>-0.044893</td>
      <td>-0.011858</td>
      <td>-0.030585</td>
      <td>-0.073138</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>tax_liens</th>
      <td>0.015755</td>
      <td>0.015755</td>
      <td>0.015733</td>
      <td>0.015073</td>
      <td>0.021572</td>
      <td>0.009393</td>
      <td>0.039479</td>
      <td>-0.006130</td>
      <td>-0.035036</td>
      <td>0.004700</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>tot_hi_cred_lim</th>
      <td>0.332977</td>
      <td>0.332977</td>
      <td>0.332989</td>
      <td>-0.097597</td>
      <td>0.293538</td>
      <td>0.130913</td>
      <td>0.411099</td>
      <td>0.025668</td>
      <td>0.034374</td>
      <td>0.057963</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>total_bal_ex_mort</th>
      <td>0.270811</td>
      <td>0.270811</td>
      <td>0.270801</td>
      <td>0.021583</td>
      <td>0.252577</td>
      <td>0.018146</td>
      <td>0.336079</td>
      <td>0.008061</td>
      <td>0.262438</td>
      <td>0.038416</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>total_bc_limit</th>
      <td>0.372026</td>
      <td>0.372026</td>
      <td>0.371955</td>
      <td>-0.185556</td>
      <td>0.334458</td>
      <td>0.066976</td>
      <td>0.276032</td>
      <td>0.030958</td>
      <td>0.049492</td>
      <td>-0.082722</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>total_il_high_credit_limit</th>
      <td>0.203510</td>
      <td>0.203510</td>
      <td>0.203500</td>
      <td>0.013691</td>
      <td>0.186510</td>
      <td>0.008036</td>
      <td>0.283449</td>
      <td>0.004152</td>
      <td>0.296579</td>
      <td>0.066601</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>deferral_term</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>hardship_amount</th>
      <td>0.827591</td>
      <td>0.827591</td>
      <td>0.827475</td>
      <td>0.639130</td>
      <td>0.781581</td>
      <td>0.035424</td>
      <td>0.303210</td>
      <td>-0.010632</td>
      <td>0.083784</td>
      <td>-0.006715</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>hardship_length</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>hardship_dpd</th>
      <td>0.034837</td>
      <td>0.034837</td>
      <td>0.034631</td>
      <td>0.142229</td>
      <td>0.051037</td>
      <td>-0.048772</td>
      <td>-0.007155</td>
      <td>-0.288293</td>
      <td>0.017312</td>
      <td>0.049492</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.062390</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>orig_projected_additional_accrued_interest</th>
      <td>0.829480</td>
      <td>0.829480</td>
      <td>0.829360</td>
      <td>0.633758</td>
      <td>0.783187</td>
      <td>0.042427</td>
      <td>0.312575</td>
      <td>-0.017930</td>
      <td>0.079196</td>
      <td>0.001766</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>0.078457</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>hardship_payoff_balance_amount</th>
      <td>0.958502</td>
      <td>0.958502</td>
      <td>0.958435</td>
      <td>0.324071</td>
      <td>0.862753</td>
      <td>0.072547</td>
      <td>0.432046</td>
      <td>0.041810</td>
      <td>0.003633</td>
      <td>0.001576</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.900165</td>
      <td>NaN</td>
      <td>0.011203</td>
      <td>0.901693</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>hardship_last_payment_amount</th>
      <td>0.489119</td>
      <td>0.489119</td>
      <td>0.489032</td>
      <td>0.190434</td>
      <td>0.493799</td>
      <td>0.002568</td>
      <td>0.284256</td>
      <td>0.004410</td>
      <td>0.002036</td>
      <td>0.028093</td>
      <td>...</td>
      <td>NaN</td>
      <td>0.405107</td>
      <td>NaN</td>
      <td>0.070852</td>
      <td>0.394808</td>
      <td>0.459846</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>settlement_amount</th>
      <td>0.917878</td>
      <td>0.917878</td>
      <td>0.917875</td>
      <td>0.388751</td>
      <td>0.814741</td>
      <td>0.079187</td>
      <td>0.492165</td>
      <td>0.068815</td>
      <td>0.082613</td>
      <td>0.059831</td>
      <td>...</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>-1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>settlement_percentage</th>
      <td>-0.226310</td>
      <td>-0.226310</td>
      <td>-0.226299</td>
      <td>-0.148911</td>
      <td>-0.224885</td>
      <td>-0.095090</td>
      <td>-0.051463</td>
      <td>0.362477</td>
      <td>-0.097457</td>
      <td>-0.051380</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.018362</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>settlement_term</th>
      <td>0.575292</td>
      <td>0.575292</td>
      <td>0.575291</td>
      <td>0.286584</td>
      <td>0.507058</td>
      <td>0.111925</td>
      <td>0.226042</td>
      <td>-0.263670</td>
      <td>0.318564</td>
      <td>0.026849</td>
      <td>...</td>
      <td>NaN</td>
      <td>-1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1.000000</td>
      <td>-1.000000</td>
      <td>1.0</td>
      <td>0.563137</td>
      <td>-0.050615</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>84 rows × 84 columns</p>
</div>




```python
cor.stack()
```




    loan_amnt        loan_amnt                                     0.000000
                     funded_amnt                                   0.000000
                     funded_amnt_inv                               0.000000
                     int_rate                                      0.000000
                     installment                                   0.000000
                     emp_length                                    0.000000
                     annual_inc                                    0.000000
                     loan_status                                   0.000000
                     dti                                           0.000000
                     delinq_2yrs                                   0.000000
                     inq_last_6mths                                0.000000
                     mths_since_last_delinq                        0.000000
                     mths_since_last_record                        0.000000
                     open_acc                                      0.000000
                     pub_rec                                       0.000000
                     revol_bal                                     0.000000
                     total_acc                                     0.000000
                     collections_12_mths_ex_med                    0.000000
                     mths_since_last_major_derog                   0.000000
                     acc_now_delinq                                0.000000
                     tot_coll_amt                                  0.000000
                     tot_cur_bal                                   0.000000
                     open_acc_6m                                   0.000000
                     open_act_il                                   0.000000
                     open_il_12m                                   0.000000
                     open_il_24m                                   0.000000
                     mths_since_rcnt_il                            0.000000
                     total_bal_il                                  0.000000
                     il_util                                       0.000000
                     open_rv_12m                                   0.000000
                                                                     ...   
    settlement_term  mths_since_recent_inq                         0.030340
                     mths_since_recent_revol_delinq                0.103703
                     num_accts_ever_120_pd                         0.027002
                     num_actv_bc_tl                                0.181307
                     num_actv_rev_tl                               0.220559
                     num_bc_sats                                   0.162077
                     num_bc_tl                                     0.178956
                     num_il_tl                                     0.114414
                     num_op_rev_tl                                 0.166385
                     num_rev_accts                                 0.197364
                     num_rev_tl_bal_gt_0                           0.247732
                     num_sats                                      0.193849
                     num_tl_30dpd                                 -0.096457
                     num_tl_90g_dpd_24m                            0.068411
                     num_tl_op_past_12m                            0.110434
                     pct_tl_nvr_dlq                                0.121179
                     percent_bc_gt_75                              0.027080
                     pub_rec_bankruptcies                         -0.080801
                     tax_liens                                     0.039254
                     tot_hi_cred_lim                               0.213624
                     total_bal_ex_mort                             0.219922
                     total_bc_limit                                0.184810
                     total_il_high_credit_limit                    0.218572
                     hardship_amount                              -1.000000
                     orig_projected_additional_accrued_interest   -1.000000
                     hardship_payoff_balance_amount               -1.000000
                     hardship_last_payment_amount                  1.000000
                     settlement_amount                             0.563137
                     settlement_percentage                        -0.050615
                     settlement_term                               0.000000
    Length: 6876, dtype: float64




```python
cor[(cor>0.55) | (cor<-0.55)].stack()
```




    funded_amnt                                 loan_amnt                                     1.000000
    funded_amnt_inv                             loan_amnt                                     0.999993
                                                funded_amnt                                   0.999993
    installment                                 loan_amnt                                     0.953658
                                                funded_amnt                                   0.953658
                                                funded_amnt_inv                               0.953570
    mths_since_last_delinq                      delinq_2yrs                                  -0.551436
    total_acc                                   open_acc                                      0.723418
    mths_since_last_major_derog                 mths_since_last_delinq                        0.691480
    open_il_24m                                 open_il_12m                                   0.758545
    total_bal_il                                open_act_il                                   0.567060
    open_rv_12m                                 open_acc_6m                                   0.623141
    open_rv_24m                                 open_rv_12m                                   0.776720
    all_util                                    il_util                                       0.597521
    total_rev_hi_lim                            revol_bal                                     0.815654
    inq_last_12m                                inq_fi                                        0.559390
    acc_open_past_24mths                        open_acc_6m                                   0.553789
                                                open_il_24m                                   0.570438
                                                open_rv_12m                                   0.659344
                                                open_rv_24m                                   0.847494
    avg_cur_bal                                 tot_cur_bal                                   0.828143
    bc_open_to_buy                              total_rev_hi_lim                              0.624163
    bc_util                                     all_util                                      0.572605
    mo_sin_rcnt_tl                              mo_sin_rcnt_rev_tl_op                         0.604777
    mths_since_recent_bc                        mo_sin_rcnt_rev_tl_op                         0.615410
    mths_since_recent_bc_dlq                    mths_since_last_delinq                        0.750604
                                                mths_since_last_major_derog                   0.560775
    mths_since_recent_revol_delinq              mths_since_last_delinq                        0.852881
                                                mths_since_recent_bc_dlq                      0.879875
    num_actv_bc_tl                              open_acc                                      0.553697
                                                                                                ...   
    hardship_amount                             installment                                   0.781581
    orig_projected_additional_accrued_interest  loan_amnt                                     0.829480
                                                funded_amnt                                   0.829480
                                                funded_amnt_inv                               0.829360
                                                int_rate                                      0.633758
                                                installment                                   0.783187
                                                hardship_amount                               1.000000
    hardship_payoff_balance_amount              loan_amnt                                     0.958502
                                                funded_amnt                                   0.958502
                                                funded_amnt_inv                               0.958435
                                                installment                                   0.862753
                                                hardship_amount                               0.900165
                                                orig_projected_additional_accrued_interest    0.901693
    settlement_amount                           loan_amnt                                     0.917878
                                                funded_amnt                                   0.917878
                                                funded_amnt_inv                               0.917875
                                                installment                                   0.814741
                                                mths_since_last_record                       -0.600571
                                                hardship_amount                               1.000000
                                                orig_projected_additional_accrued_interest    1.000000
                                                hardship_payoff_balance_amount                1.000000
                                                hardship_last_payment_amount                 -1.000000
    settlement_term                             loan_amnt                                     0.575292
                                                funded_amnt                                   0.575292
                                                funded_amnt_inv                               0.575291
                                                hardship_amount                              -1.000000
                                                orig_projected_additional_accrued_interest   -1.000000
                                                hardship_payoff_balance_amount               -1.000000
                                                hardship_last_payment_amount                  1.000000
                                                settlement_amount                             0.563137
    Length: 125, dtype: float64




```python
df.drop(['funded_amnt','funded_amnt_inv', 'installment'], axis=1, inplace=True)
```

### 建立模型


```python
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
from sklearn.preprocessing import OneHotEncoder
```


```python
Y = df.loan_status
X = df.drop('loan_status',1,inplace=False)
```


```python
print(Y.shape)
print(X.shape)
```

    (86770,)
    (86770, 89)
    


```python
X = pd.get_dummies(X)
```


```python
print(X.shape)
```

    (86770, 100)
    


```python
X.fillna(0.0,inplace=True)
X.fillna(0,inplace=True)
```


```python
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.3, random_state=666)
```


```python
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
```

    (60739, 100)
    (60739,)
    (26031, 100)
    (26031,)
    


```python
print (y_train.value_counts())
print (y_test.value_counts())
```

    1.0    59148
    0.0     1591
    Name: loan_status, dtype: int64
    1.0    25348
    0.0      683
    Name: loan_status, dtype: int64
    


```python
param_grid = {'learning_rate': np.linspace(0.01,0.5,5),
              'max_depth': [i for i in range(2,6)],
              'min_samples_split': [50,100],
              'n_estimators': [100,200]
              }
```


```python
grid_search = GridSearchCV(ensemble.GradientBoostingRegressor(),
                   param_grid, n_jobs=4, refit=True,verbose=1)
```


```python
%%time
grid_search.fit(x_train,y_train)
bst_clf = grid_search.best_estimator_
```

    Fitting 3 folds for each of 80 candidates, totalling 240 fits
    

    [Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:  5.3min
    [Parallel(n_jobs=4)]: Done 192 tasks      | elapsed: 31.5min
    [Parallel(n_jobs=4)]: Done 240 out of 240 | elapsed: 40.6min finished
    

    Wall time: 41min 17s
    


```python
%%time
bst_clf.fit(x_train, y_train)
```

    Wall time: 43.2 s
    




    GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                 learning_rate=0.01, loss='ls', max_depth=4, max_features=None,
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, min_samples_leaf=1,
                 min_samples_split=100, min_weight_fraction_leaf=0.0,
                 n_estimators=200, presort='auto', random_state=None,
                 subsample=1.0, verbose=0, warm_start=False)




```python
bst_clf.score(x_test,y_test)
```




    0.07240211533402607




```python
def compute_ks(data):

    sorted_list = data.sort_values(['predict'], ascending=[True])

    total_bad = sorted_list['label'].sum(axis=None, skipna=None, level=None, numeric_only=None) / 3
    total_good = sorted_list.shape[0] - total_bad

    max_ks = 0.0
    good_count = 0.0
    bad_count = 0.0
    for index, row in sorted_list.iterrows():
        if row['label'] == 3:
            bad_count += 1.0
        else:
            good_count += 1.0

        val = bad_count/total_bad - good_count/total_good
        max_ks = max(max_ks, val)

    return max_ks
```


```python
test_pd = pd.DataFrame()
test_pd['predict'] = bst_clf.predict(x_test)
test_pd['label'] = y_test
print( compute_ks(test_pd[['label','predict']]))
```

    0.0
    


```python
feature_importance = bst_clf.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())

indices = np.argsort(feature_importance)[-10:]
plt.barh(np.arange(10), feature_importance[indices],color='dodgerblue',alpha=.4)
plt.yticks(np.arange(10 + 0.25), np.array(X.columns)[indices])
_ = plt.xlabel('Relative importance'), plt.title('Top Ten Important Variables')
```


![png](output_73_0.png)

