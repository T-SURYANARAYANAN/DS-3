## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
     Name: SURYANARAYANAN T
     Register Number: 212224040341
```
     
    import pandas as pd
    df=pd.read_csv("/content/Encoding Data.csv")
    df
  ![image](https://github.com/user-attachments/assets/48863bbf-ff9b-4f61-bf2a-ecc4e22c94c4)

    from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
    pm=['Hot','Warm','Cold']
    e1=OrdinalEncoder(categories=[pm])
    e1.fit_transform(df[["ord_2"]])
![image](https://github.com/user-attachments/assets/78be10b0-6598-44b8-9a8a-8bd5f36f5b50)

    df['bo2']=e1.fit_transform(df[["ord_2"]])
    df
![image](https://github.com/user-attachments/assets/905d38e9-7223-4196-8670-44a33a5da794)

    le=LabelEncoder()
    dfc=df.copy()
    dfc['ord_2']=le.fit_transform(dfc['ord_2'])
    dfc
![image](https://github.com/user-attachments/assets/e4b170e4-0f5e-41b7-ae8d-82ceed4f36f4)

    from sklearn.preprocessing import OneHotEncoder
    ohe=OneHotEncoder(sparse=False)
    df2=df.copy()
    enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))

![image](https://github.com/user-attachments/assets/4f94c56d-389e-4d4c-95ff-3f38bfa1d095)

    df2=pd.concat([df2,enc],axis=1)
    df2 
![image](https://github.com/user-attachments/assets/5a1e1ffa-2aec-4375-bc37-469ffc5a37e6)

    pd.get_dummies(df2,columns=["nom_0"])
![image](https://github.com/user-attachments/assets/276d3492-f8f5-407d-ac5f-9aab8297105d)

    pip install --upgrade category_encoders

![image](https://github.com/user-attachments/assets/0be42753-d2f4-4b94-96df-ba30aa143c6e)

    from category_encoders import BinaryEncoder
    df=pd.read_csv("/content/data.csv")
    df

![image](https://github.com/user-attachments/assets/0f668e5e-a2f8-4f26-b233-66dfe2762a02)

    be=BinaryEncoder()
    nd=be.fit_transform(df['Ord_2'])
    dfb=pd.concat([df,nd],axis=1)
    dfb1=df.copy()
    dfb 

![image](https://github.com/user-attachments/assets/bd27592b-ab81-4888-b368-69eec0e91a7d)

    from category_encoders import TargetEncoder
    te=TargetEncoder()
    CC=df.copy()
    new=te.fit_transform(X=CC["City"],y=CC["Target"])
    CC=pd.concat([CC,new],axis=1)
    CC
    
![image](https://github.com/user-attachments/assets/16c3bd83-f8cd-42ec-8171-dcd2d61d9c93)

    import pandas as pd
    from scipy import stats
    import numpy as np
    df=pd.read_csv("/content/Data_to_Transform.csv")
    df

![image](https://github.com/user-attachments/assets/f58c52b5-66b3-41c0-aaf0-a948f5d71510)

    df.skew() 
![image](https://github.com/user-attachments/assets/8c5f11d2-e594-498d-a1fc-b985f51fe2be)

    np.log(df["Highly Positive Skew"])

![image](https://github.com/user-attachments/assets/963e7f55-11a3-46f8-8aa1-48b0402c9cde)

    np.reciprocal(df["Moderate Positive Skew"])

![image](https://github.com/user-attachments/assets/bc108073-6348-4e3a-aed0-efe4827e0221)

    np.sqrt(df["Highly Positive Skew"])

![image](https://github.com/user-attachments/assets/22a0f5d5-aa9c-4a68-af8a-1fd304daaf7e)

    np.square(df["Highly Positive Skew"])

![image](https://github.com/user-attachments/assets/b21c2099-233c-469c-8039-7965f2ffc06f)

    df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
    df

![image](https://github.com/user-attachments/assets/969b9698-d5d4-4712-8e55-af175cab2539)

    df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
    
![image](https://github.com/user-attachments/assets/e25a3801-1597-400f-b371-2ccee1e4777b)

    import seaborn as sns
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    sm.qqplot(df["Moderate Negative Skew"],line='45')
    plt.show()

![image](https://github.com/user-attachments/assets/385a0e83-e7ea-4738-a8f1-c258e30066bb)

    sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line='45')
    plt.show()

![image](https://github.com/user-attachments/assets/aefc8715-51be-495b-8fe5-50e57b28a168)

    from sklearn.preprocessing import QuantileTransformer
    qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

    df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

    sm.qqplot(df["Moderate Negative Skew"],line='45')
    plt.show()

![image](https://github.com/user-attachments/assets/1525fed6-dd71-49b4-b8b8-9edaac7be490)

    df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
    sm.qqplot(df["Highly Negative Skew"],line='45')
    plt.show()   
![image](https://github.com/user-attachments/assets/21714cee-39af-4449-9258-7aed317bf833)

    sm.qqplot(df["Highly Negative Skew_1"],line='45')
    plt.show()
![image](https://github.com/user-attachments/assets/1d5aeb23-6034-4f18-aad9-c1e3885e668c)

    sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
    plt.show()
![image](https://github.com/user-attachments/assets/853676c3-b9ac-4f5e-ad8f-78cdd6c55f9b)

# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.       

       
