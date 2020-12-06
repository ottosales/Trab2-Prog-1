module Algorithms(
    genRandom,
    getFoldsIndex,
    classifyAllItemsCV,
    classifyAllCentroidsCV,
    classifyAllKNNCV,
    accWZScore,
    stdDev,
    confusionTable,
    getTestList,
    getClassList,
    printConfTable,
    makeTuple
)
where

import System.Random ( mkStdGen, Random(randomRs) )
import Data.List as List (transpose, delete, elemIndex)
import Data.Maybe as Maybe (fromMaybe)

type DataTuple = ([Double], String)

-- | creates a tuple
makeTuple :: a -> b -> (a, b)
makeTuple x y = (x, y)

-- | teacher's function to remove duplicate values from a list
removeDup :: Eq a => [a] -> [a]
removeDup l = removeD l []
   where
     removeD [] _ = []
     removeD (x:xs) ls
        | x `elem` ls = removeD xs ls
        | otherwise = x: removeD xs (x:ls)

-- | generates a list of random numbers given a seed, a size and a limit to the values
genRandom :: Int -> Int -> Int -> [Int]
genRandom seed testSize limit = take testSize (removeDup (randomRs (0,limit-1) $ mkStdGen seed:: [Int]))

-- | gets a list of index considering the rest of values / nFolds equal to zero
getWholeFoldGroups :: [Int] -> Int -> Int ->[[Int]]
getWholeFoldGroups _ _ 0 = []
getWholeFoldGroups randomList foldSize n = take foldSize randomList : getWholeFoldGroups (drop foldSize randomList) foldSize (n-1)

-- | insert the remaining index values in the list
insertRemaining :: [[Int]] -> [Int] -> [[Int]]
insertRemaining wholeFoldList remainings = [wholeFoldList !! x ++ [remainings !! x] | x <- [0..length remainings - 1]] ++ drop (length remainings) wholeFoldList

-- | gets a list of index values 
getFoldsIndex :: [Int] -> Int -> [[Int]]
getFoldsIndex randomList nFolds = insertRemaining (getWholeFoldGroups randomList (length randomList `div` nFolds) nFolds) (drop (nFolds * (length randomList `div` nFolds)) randomList)

-- | calculates a distance between two points
calcDist :: DataTuple -> DataTuple -> Double
calcDist (xs, _) (ys, _) = sqrt . sum $ [uncurry (-) z ** 2 | z <- zip xs ys]

-- | function that tries to guess a point's class with a given training group
guessClassPoint :: [DataTuple] -> DataTuple -> DataTuple
guessClassPoint trainingGroup point = makeTuple (fst point) (snd (trainingGroup !! Maybe.fromMaybe (-1) (elemIndex (minimum list) list))) 
    where
        list = [calcDist x point | x <- trainingGroup]

getKSmallestInList :: [Double] -> [DataTuple] -> Int -> [(Double, DataTuple)]
getKSmallestInList _ _ 0 = []
getKSmallestInList list trainingGroup k = makeTuple (minimum list) picked : getKSmallestInList (delete (minimum list) list) (delete picked trainingGroup) (k-1)
    where
        picked = trainingGroup !! Maybe.fromMaybe (-1) (elemIndex (minimum list) list)

returnCountFromClass :: [(Double, DataTuple)] -> String -> Int -> Int
returnCountFromClass [] _ value = value
returnCountFromClass (x:xs) className value =
    if snd (snd x) == className then
        returnCountFromClass xs className (succ value)
        else returnCountFromClass xs className value

-- | get a list of all classes in a given list
getClassListAux :: Ord a1 => [(a3, (a2, a1))] -> [a1]
getClassListAux dataList = removeDup [snd (snd x) | x <- dataList]

getAllClassesCount :: [(Double, DataTuple)] -> [Int]
getAllClassesCount kSmallestInList = [returnCountFromClass kSmallestInList x 0 | x <- getClassListAux kSmallestInList]

calcMeanDistForClass :: [(Double, DataTuple)] -> String -> Double -> Int -> Double
calcMeanDistForClass [] _ value nOccurrences = value / fromIntegral nOccurrences
calcMeanDistForClass (x:xs) className value nOccurrences =
    if snd (snd x) == className then
        calcMeanDistForClass xs className (value + fst x) nOccurrences
        else calcMeanDistForClass xs className value nOccurrences

resolveKNN :: [(Double, DataTuple)] -> [String] -> [Int] -> String
resolveKNN kSmallestInList classList countList = classList !! Maybe.fromMaybe (-1) (elemIndex (minimum meanList) meanList)
    where
        meanList = [calcMeanDistForClass kSmallestInList (classList !! x) 0 (countList !! x) | x <- [0..length classList - 1]]

guessClassPointKNN :: [DataTuple] -> DataTuple -> Int -> DataTuple
guessClassPointKNN trainingGroup point k =
    let kSmallestInList = getKSmallestInList [calcDist x point | x <- trainingGroup] trainingGroup k
        classList = getClassListAux kSmallestInList
        countList = getAllClassesCount kSmallestInList
        in if length (filter (==maximum countList) countList) == 1 then
            makeTuple (fst point) (classList !! Maybe.fromMaybe (-1) (elemIndex (maximum countList) countList))
            else makeTuple (fst point) (resolveKNN kSmallestInList classList countList)

classifyAllItemsKNN :: [DataTuple] -> [DataTuple] -> Int -> [DataTuple]
classifyAllItemsKNN trainingGroup testGroup k = [guessClassPointKNN trainingGroup x k | x <- testGroup]

-- | classifies all points of a given test group
classifyAllItems :: [DataTuple] -> [DataTuple] -> [DataTuple]
classifyAllItems trainingGroup testGroup = [ guessClassPoint trainingGroup x | x <- testGroup]

-- | classifies all points using cross validation
classifyAllItemsCV :: [DataTuple] -> [[Int]] -> [[DataTuple]]
classifyAllItemsCV dataList foldIndex = [classifyAllItems (zScoredTraining x) (zScoredTestList x) | x <- foldIndex]
    where
        zScoredTraining x = standardizeValues (getTrainingList dataList x) (applyZScore (getTrainingList dataList x) (calcMeanAndStdDevFromList (getTrainingList dataList x))) 
        zScoredTestList x = standardizeValues (getTestList dataList x) (applyZScore (getTestList dataList x) (calcMeanAndStdDevFromList (getTrainingList dataList x)))

-- | classifies all centroids using cross validation
classifyAllCentroidsCV :: [DataTuple] -> [[Int]] -> [[DataTuple]]
classifyAllCentroidsCV dataList foldIndex = [classifyAllItems (calcAllCentroids (zScoredTraining x)) (zScoredTestList x) | x <- foldIndex]
    where
        zScoredTraining x = standardizeValues (getTrainingList dataList x) (applyZScore (getTrainingList dataList x) (calcMeanAndStdDevFromList (getTrainingList dataList x))) 
        zScoredTestList x = standardizeValues (getTestList dataList x) (applyZScore (getTestList dataList x) (calcMeanAndStdDevFromList (getTrainingList dataList x)))

-- | classifies the K Nearest Neighbours using cross validation
classifyAllKNNCV :: [DataTuple] -> [[Int]] -> Int -> [[DataTuple]]
classifyAllKNNCV dataList foldIndex k = [classifyAllItemsKNN (zScoredTraining x) (zScoredTestList x) k | x <- foldIndex]
    where
        zScoredTraining x = standardizeValues (getTrainingList dataList x) (applyZScore (getTrainingList dataList x) (calcMeanAndStdDevFromList (getTrainingList dataList x))) 
        zScoredTestList x = standardizeValues (getTestList dataList x) (applyZScore (getTestList dataList x) (calcMeanAndStdDevFromList (getTrainingList dataList x)))

-- | gets the test list
getTestList :: [DataTuple] -> [Int] -> [DataTuple]
getTestList dataList testList = [dataList !! x | x <- testList]

-- | gets the training list
getTrainingList :: [DataTuple] -> [Int] -> [DataTuple]
getTrainingList dataList testList = [dataList !! x | x <- [0..length dataList - 1], x `notElem` testList]

-- | counts the number off total right guesses the program did
rightCount :: [DataTuple] -> [DataTuple] -> Int -> Int
rightCount [] [] value = value
rightCount (x:xs) (y:ys) value =
    if snd x == snd y then
        rightCount xs ys (succ value)
        else rightCount xs ys value

-- | calculates the accuracy of the program
calcAccuracy :: Int -> Int -> Double
calcAccuracy right total = fromIntegral (100 * right) / fromIntegral total

-- | calculates the accuracy of each fold and returns it's average value
accWZScore :: [DataTuple] -> [[DataTuple]] -> [[Int]] -> Int -> Double
accWZScore dataList classifiedItems foldIndex nFolds = sum [calcAccuracy (rightCount (regularTestList x) (classifiedItems !! Maybe.fromMaybe (-1) (elemIndex x foldIndex)) 0) (length x) | x <- foldIndex] / fromIntegral nFolds
    where
        regularTestList x = getTestList dataList x 
-- | calculates the standard deviation of the accuracies
stdDev :: [DataTuple] -> [[DataTuple]] -> [[Int]] -> Int -> Double
stdDev dataList classifiedItems foldIndex nFolds = sqrt (sum [(x - mean)**2 | x <- eachAccuracy] / fromIntegral (length eachAccuracy))
    where
        eachAccuracy = [calcAccuracy (rightCount (regularTestList x) (classifiedItems !! Maybe.fromMaybe (-1) (elemIndex x foldIndex)) 0) (length x) | x <- foldIndex]
        mean = accWZScore dataList classifiedItems foldIndex nFolds
        regularTestList x = getTestList dataList x 

-- | gets all elements of a given class
getAllElementsInClass :: [DataTuple] -> String -> [[Double]]
getAllElementsInClass dataList className = [ fst x | x <- dataList, snd x == className]

-- | calculates a centroid of given points
calcCentroid :: [[Double]] -> [Double]
calcCentroid [] = []
calcCentroid [element] = element
calcCentroid (element : elements) = zipWith (+) element (calcCentroid elements)

-- | get a list of all classes in a given list
getClassList :: Ord a1 => [(a2, a1)] -> [a1]
getClassList dataList = removeDup [snd x | x <- dataList]

-- | calculates all centroids
calcAllCentroids :: [DataTuple] -> [DataTuple]
calcAllCentroids dataList = [makeTuple (map (/ fromIntegral (length (getAllElementsInClass dataList x))) (calcCentroid (getAllElementsInClass dataList x))) x | x <- getClassList dataList]

-- | informs the number of times the program guessed a class "guessedClass" and the right class was "rightClass", usefull with the function below
calcGuesses :: [DataTuple] -> [DataTuple] -> String -> String -> Int -> Int
calcGuesses [] [] _ _ value = value
calcGuesses  (x:xs) (y:ys) guessedClass rightClass value =
    if snd x == guessedClass && snd y == rightClass then
        calcGuesses xs ys guessedClass rightClass (succ value)
        else calcGuesses xs ys guessedClass rightClass value

-- | calculates the whole confusion table (aka confusion matrix)
confusionTable :: [DataTuple] -> [DataTuple] -> [String] -> [[Int]]
confusionTable guessedGroup testGroup classes = [[ calcGuesses guessedGroup testGroup x y 0 | y <- classes ] | x <- classes]

-- | right align a number, and transforms it into a string
fixNumberPosition :: Int -> String
fixNumberPosition x
    | x > 9    = show x
    | otherwise = " " ++ show x

-- | creates a line to be shown in the confusion table
createConfTableLine :: [String] -> [Char] -> [Char]
createConfTableLine [] line = line
createConfTableLine [x] _ = x
createConfTableLine (x:xs) line = line ++ x ++ ", " ++ createConfTableLine xs line

-- | displays the confusion table
printConfTable :: [[Int]] -> String
printConfTable confTable = unlines [" " ++ createConfTableLine [fixNumberPosition x | x <- y] "" | y <- confTable]

-- | calculates the zscore
zscore :: Double -> Double -> Double -> Double
zscore mean stdDev value = (value - mean) / stdDev

-- | calculates the mean value of a list of doubles
calcMean :: [Double] -> Double
calcMean list = sum list / fromIntegral (length list)

-- | calculates the standard deviation of a list of doubles
calcStdDev :: [Double] -> Double -> Double
calcStdDev list mean = sqrt (sum [(x - mean)**2 | x <- list] / fromIntegral (length list))

-- | transposes a DataTuple using the first value of the tuple
transposeDataList :: [DataTuple] -> [[Double]]
transposeDataList dataList = transpose [fst x | x <- dataList]

-- | calculates the mean and the standard deviation of a list of DataTuples
calcMeanAndStdDevFromList :: [DataTuple] -> [(Double, Double)]
calcMeanAndStdDevFromList dataList = [makeTuple (calcMean x) (calcStdDev x (calcMean x)) | x <- transposeDataList dataList]

-- | applies the standardization to the list of data
applyZScore :: [DataTuple] -> [(Double, Double)] -> [[Double]]
applyZScore dataList meanAndSD = [map (uncurry zscore (meanAndSD !! x)) (transposedDL !! x) | x <- [0..length transposedDL -1]]
    where
        transposedDL = transposeDataList dataList

-- | given a datalist and it's transposed zscored list, returns the DataTuple zscored
standardizeValues :: [DataTuple] -> [[Double]] -> [([Double], String)]
standardizeValues dataList transposedList = [makeTuple (untransposedList !! x) (snd (dataList !! x))| x <- [0..length dataList -1]]
    where
        untransposedList = transpose transposedList