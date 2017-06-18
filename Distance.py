import math

def similarityScore(person1, person2):

    bothRated = 0
    for item, score in zip(person1.indices, person2.data):
        if item in person2.indices:
            for it, sc in zip(person2.indices, person2.data):
                if it == item:
                    bothRated += pow(score-sc,2)
                    break
    return bothRated

def similarityScore2(person1, person2):

    bothRated = 0
    for item, score in zip(person1.indices, person2.data):
        for it, sc in zip(person2.indices, person2.data):
             if it == item:
                bothRated += pow(score-sc,2)
                break
    return bothRated

def pearsonCorrelation(person1, person2):

    person1RatingSum = 0
    person2RatingSum = 0
    person1RatingsSqSum = 0
    person2RatingsSqSum = 0
    productRatingSum = 0
    numberOfRatings = 0
    for item, score in zip(person1.indices, person1.data):
        for it, sc in zip(person2.indices, person2.data):
            if it == item:
                numberOfRatings += 1
                person1RatingSum += score
                person2RatingSum += sc
                person1RatingsSqSum += pow(score, 2)
                person2RatingsSqSum += pow(sc, 2)
                productRatingSum += score * sc
                break

    if numberOfRatings == 0:
        return -1

    numerator = productRatingSum - (person1RatingSum * person2RatingSum / numberOfRatings)
    denominator = math.sqrt(
                                (person1RatingsSqSum - pow(person1RatingSum, 2) / numberOfRatings)
                                * (person2RatingsSqSum - pow(person2RatingSum, 2) / numberOfRatings)
                            )
    if denominator == 0:
        return -1

    pearsonScore = numerator / denominator
    return pearsonScore
