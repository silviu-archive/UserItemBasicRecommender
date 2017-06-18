import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
import joblib

from Distance import similarityScore, similarityScore2, pearsonCorrelation
from ReadData import readData

import timeit
import operator

def main():

    #Parameters
    testData = 1
    if testData == 1:
        fileAddress = 'ClothingTestData.csv'
        matrixAddress = 'Test.pkl'
    else:
        fileAddress = 'ClothingShoesJewelsRatingShort.csv'
        matrixAddress = 'Original.pkl'

    #State if you need to create the matrix again or not
    matrixCreation = 1
    if matrixCreation == 0:
        ratingsMatrix = joblib.load(matrixAddress)
        userArray = None
    else:
        ratingsMatrix, userArray = readData(fileAddress, matrixAddress)

    #SelectUser based on index
    chosenUser = 100
    if userArray is not None:
        chosenUserID = userArray[chosenUser]
    else:
        chosenUserID = chosenUser

    #Get cooMatrix and csrMatrix for data handling - cooMatrix is only used for delimiting unique users
    ratingsMatrixCOO = ratingsMatrix.tocoo()
    ratingsMatrixCSR = ratingsMatrix.tocsr()
    del ratingsMatrix

    #Testing distance measures
    '''for p1 in ratingsMatrixCOO.row:
        counter = timeit.default_timer()
        for p2 in ratingsMatrixCOO.row:
            person1 = ratingsMatrixCSR[p1, :]
            person2 = ratingsMatrixCSR[p2, :]

            #c1 = timeit.default_timer()
            #score = similarityScore(person1, person2)
            #print(1, timeit.default_timer() - c1)
            #c1 = timeit.default_timer()
            #score2 = similarityScore2(person1, person2)
            #print(2, timeit.default_timer() - c1)
            c1 = timeit.default_timer()
            score3 = pearsonCorrelation(person1, person2)
            print('Pearson', timeit.default_timer() - c1)
        print('One comparison', timeit.default_timer()-counter)'''

    #Get the csr matrix for our user
    person1 = ratingsMatrixCSR[chosenUser, :]

    #Testing score calculations
    '''
    similarPersons = {}
    #Iterate over all the other existing users
    for p2 in np.unique(ratingsMatrixCOO.row):
        #Get the csrMatrix of a user
        person2 = ratingsMatrixCSR[p2, :]
        #Calculate the pearson correlation between our target user and another user
        score = pearsonCorrelation(person1, person2)
        if score >= 0:
            similarPersons[p2] = score
    sortedPersons = sorted(similarPersons.items(), key=operator.itemgetter(1), reverse=True)
    '''

    #Starting rating process
    totalRating = {}
    similaritySums = {}

    #Go through all other users
    for p2 in np.unique(ratingsMatrixCOO.row):
        #Select their CSR matrix
        person2 = ratingsMatrixCSR[p2, :]

        #Skip over identical users (including itself)
        if p2 == chosenUser:
            continue
        #Calculate pearson score ignore if 0 or below
        score = pearsonCorrelation(person1, person2)
        if score <= 0:
            continue

        #Iterate over all the items p2 rated
        for item in person2.indices:
            #If p1 did not rate an item
            if item not in person1.indices:
                #Score them
                totalRating.setdefault(item, 0)
                totalRating[item] += person2[:,item].data[0] * score
                similaritySums.setdefault(item, 0)
                similaritySums[item] += score

    #Create a list of (score, item) tuples
    rankings = [(total / similaritySums[item], item) for item, total in totalRating.items()]
    rankings = sorted(rankings, reverse=True)

    #Create the pure recommendations list
    recommendations = [recommendedItem for score, recommendedItem in rankings]

    print(rankings)
    print('Recommended items for user %s: ' % (chosenUserID), recommendations)





main()