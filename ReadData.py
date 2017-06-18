import pandas as pd
from scipy.sparse import lil_matrix
import timeit
from joblib import dump


def readData(fileaddress, outputAddress):

    #Read CSV into DataFrame
    df = pd.read_csv(fileaddress, header=None)
    df.columns = ['User', 'Product', 'Rating', 'Timestamp']
    df.astype(dtype = {'User': str, 'Product': str, 'Rating': int, 'Timestamp': int})
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s').dt.date
    #Will only add the most recent rating later on in the sparse matrix
    df.sort_values(by='Timestamp', inplace=True)

    #Create Dicts for Users and Products to map in sparse matrix
    userArray = df['User'].unique()
    userDict = {}
    c = 0
    for u in userArray:
        userDict[u] = c
        c+=1
    productArray = df['Product'].unique()
    productDict = {}
    c = 0
    for u in productArray:
        productDict[u] = c
        c+=1

    #Map (user, product) rating in a numpy sparse matrix
    ratingsMatrix = lil_matrix((df['User'].unique().shape[0], df['Product'].unique().shape[0]))
    start = timeit.default_timer()
    for row in df.itertuples():
        ratingsMatrix[userDict[row[1]], productDict[row[2]]] = row[3]
    end = timeit.default_timer()-start
    print('Matrix creation took: ', end)

    #Dump the matrix in a pickle file
    dump(ratingsMatrix, outputAddress)
    return ratingsMatrix, userArray