from sklearn.preprocessing import OneHotEncoder


def one_hot_encode(arr):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(arr)
    return enc.transform(arr).toarray()
