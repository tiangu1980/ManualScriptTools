import pandas as pd

def PrintDataDetail(comment1, comment2, dataobj):
    try:
        print(f"@ {comment1} {comment2} :\n{dataobj}")
    except Exception as e:
        print(f"@ {comment1} {comment2} :\n{e}")


def PrintDataDetails(comment, dataobj):
    print(f"---------------\n{comment}")
    PrintDataDetail(".....", "value", dataobj)
    PrintDataDetail(".....", "type", type(dataobj))
    try:
        shape = dataobj.shape
        PrintDataDetail(".....", "shape", shape)
    except AttributeError:
        print(f"@ ..... shape : NO Attribute")
    
    try:
        size = dataobj.size
        PrintDataDetail(".....", "size", size)
    except AttributeError:
        print(f"@ ..... size : NO Attribute")
    
    try:
        length = len(dataobj)
        PrintDataDetail(".....", "len", length)
    except TypeError:
        print(f"@ ..... len() : CAN NOT CALL")

# 创建一个示例 DataFrame
data = {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9], 'D': [10, 11, 12]}
df = pd.DataFrame(data, index=['X', 'Y', 'Z'])
df2 = pd.DataFrame(data)

PrintDataDetails("data", data)
PrintDataDetails("df = pd.DataFrame(data, index=['X', 'Y', 'Z'])", df)
PrintDataDetails("df2 = pd.DataFrame(data)", df2)
PrintDataDetails("df.loc['X']", df.loc['X'])
PrintDataDetails("df.iloc[0]", df.iloc[0])
PrintDataDetails("df.loc['X']['A']", df.loc['X']['A'])
PrintDataDetails("df.loc[['X']]", df.loc[['X']])
PrintDataDetails("df.loc[['X', 'Y']]", df.loc[['X', 'Y']])
PrintDataDetails("df.loc['X':'Y']", df.loc['X':'Y'])
PrintDataDetails("df.loc[df['A'] > 1]", df.loc[df['A'] > 1])
PrintDataDetails("df.loc[:, 'A']", df.loc[:, 'A'])
PrintDataDetails("df.loc[:, ['A', 'B']]", df.loc[:, ['A', 'B']])
PrintDataDetails("df.loc[:, 'A':'B']", df.loc[:, 'A':'B'])
PrintDataDetails("df.loc[:, df.loc['X'] > 1]", df.loc[:, df.loc['X'] > 1])

# ---------------
# data
# @ ..... value :
#             {'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9], 'D': [10, 11, 12]}
# @ ..... type :
#             <class 'dict'>
# @ ..... shape : NO Attribute
# @ ..... size : NO Attribute
# @ ..... len :
#             4
# ---------------
# df = pd.DataFrame(data, index=['X', 'Y', 'Z'])
# @ ..... value :
#                A  B  C   D
#             X  1  4  7  10
#             Y  2  5  8  11
#             Z  3  6  9  12
# @ ..... type :
#             <class 'pandas.core.frame.DataFrame'>
# @ ..... shape :
#             (3, 4)
# @ ..... size :
#             12
# @ ..... len :
#             3
# ---------------
# df2 = pd.DataFrame(data)
# @ ..... value :
#                A  B  C   D
#             0  1  4  7  10
#             1  2  5  8  11
#             2  3  6  9  12
# @ ..... type :
#             <class 'pandas.core.frame.DataFrame'>
# @ ..... shape :
#             (3, 4)
# @ ..... size :
#             12
# @ ..... len :
#             3
# ---------------
# df.loc['X']
# @ ..... value :
#             A     1
#             B     4
#             C     7
#             D    10
#             Name: X, dtype: int64
# @ ..... type :
#             <class 'pandas.core.series.Series'>
# @ ..... shape :
#             (4,)
# @ ..... size :
#             4
# @ ..... len :
#             4
# ---------------
# df.iloc[0]
# @ ..... value :
#             A     1
#             B     4
#             C     7
#             D    10
#             Name: X, dtype: int64
# @ ..... type :
#             <class 'pandas.core.series.Series'>
# @ ..... shape :
#             (4,)
# @ ..... size :
#             4
# @ ..... len :
#             4
# ---------------
# df.loc['X']['A']
# @ ..... value :
#             1
# @ ..... type :
#             <class 'numpy.int64'>
# @ ..... shape :
#             ()
# @ ..... size :
#             1
# @ ..... len() : CAN NOT CALL
# ---------------
# df.loc[['X']]
# @ ..... value :
#                A  B  C   D
#             X  1  4  7  10
# @ ..... type :
#             <class 'pandas.core.frame.DataFrame'>
# @ ..... shape :
#             (1, 4)
# @ ..... size :
#             4
# @ ..... len :
#             1
# ---------------
# df.loc[['X', 'Y']]
# @ ..... value :
#                A  B  C   D
#             X  1  4  7  10
#             Y  2  5  8  11
# @ ..... type :
#             <class 'pandas.core.frame.DataFrame'>
# @ ..... shape :
#             (2, 4)
# @ ..... size :
#             8
# @ ..... len :
#             2
# ---------------
# df.loc['X':'Y']
# @ ..... value :
#                A  B  C   D
#             X  1  4  7  10
#             Y  2  5  8  11
# @ ..... type :
#             <class 'pandas.core.frame.DataFrame'>
# @ ..... shape :
#             (2, 4)
# @ ..... size :
#             8
# @ ..... len :
#             2
# ---------------
# df.loc[df['A'] > 1]
# @ ..... value :
#                A  B  C   D
#             Y  2  5  8  11
#             Z  3  6  9  12
# @ ..... type :
#             <class 'pandas.core.frame.DataFrame'>
# @ ..... shape :
#             (2, 4)
# @ ..... size :
#             8
# @ ..... len :
#             2
# ---------------
# df.loc[:, 'A']
# @ ..... value :
#             X    1
#             Y    2
#             Z    3
#             Name: A, dtype: int64
# @ ..... type :
#             <class 'pandas.core.series.Series'>
# @ ..... shape :
#             (3,)
# @ ..... size :
#             3
# @ ..... len :
#             3
# ---------------
# df.loc[:, ['A', 'B']]
# @ ..... value :
#                A  B
#             X  1  4
#             Y  2  5
#             Z  3  6
# @ ..... type :
#             <class 'pandas.core.frame.DataFrame'>
# @ ..... shape :
#             (3, 2)
# @ ..... size :
#             6
# @ ..... len :
#             3
# ---------------
# df.loc[:, 'A':'B']
# @ ..... value :
#                A  B
#             X  1  4
#             Y  2  5
#             Z  3  6
# @ ..... type :
#             <class 'pandas.core.frame.DataFrame'>
# @ ..... shape :
#             (3, 2)
# @ ..... size :
#             6
# @ ..... len :
#             3
# ---------------
# df.loc[:, df.loc['X'] > 1]
# @ ..... value :
#                B  C   D
#             X  4  7  10
#             Y  5  8  11
#             Z  6  9  12
# @ ..... type :
#             <class 'pandas.core.frame.DataFrame'>
# @ ..... shape :
#             (3, 3)
# @ ..... size :
#             9
# @ ..... len :
#             3