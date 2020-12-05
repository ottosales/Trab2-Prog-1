import System.Random ( mkStdGen, Random(randomRs) )
import Data.List as List (sort, elemIndex)
import Data.Maybe as Maybe (fromMaybe)
import Data.Text as Text (Text, pack, unpack, splitOn)
import Text.Printf (printf)

type DataTuple = ([Double], String)

main :: IO ()
main = do
    -- input/output/folds/neighbours/reading
    dataList <- readInput
    outputFileName <- readOutput
    folds <- readFolds
    neighbours <- readNeighbours
    seed <- readSeed

    -- generating random numbers, creating and spliting lists
    let randomList = genRandom (read seed) (length dataList) (length dataList)
    let foldIndex = getFoldsIndex randomList (read folds)

    let oneNeighbour = classifyAllItemsCV dataList foldIndex
    let centroids = classifyAllCentroidsCV dataList foldIndex
    -- let kNeighbours

    print $ accuracyOneNeighbour dataList foldIndex (read folds)
    print $ accuracyCentroids dataList foldIndex (read folds)
    -- print $ accuracyKNeighbours dataList foldIndex (read folds)

    let confTableNeighbour = confusionTable oneNeighbour (getTestList dataList (concat foldIndex)) (getClassList dataList)
    let confTableCentroid = confusionTable centroids (getTestList dataList (concat foldIndex)) (getClassList dataList)

    -- printing confusion table to output file
    writeFile outputFileName ""
    appendFile outputFileName "vizinho mais próximo:\n"
    appendFile outputFileName $ printConfTable confTableNeighbour
    appendFile outputFileName "\ncentroides:\n"
    appendFile outputFileName $ printConfTable confTableCentroid
    -- appendFile outputFileName "\nk-vizinhos mais próximos:\n"
    -- appendFile outputFileName $ printConfTable confTableKN

    --all done :) end of main


-- reads the input and returns a list of tuples, with the fst element of the tuple being the point (a list of Double) and the second a string (point's class)
readInput :: IO [([Double], String)]
readInput = do
    putStrLn "Forneca o nome do arquivo de entrada:"
    fileName <- getLine
    handle <- readFile fileName
    return [lineToData handle | handle <- lines handle]

-- splits the line at the ","
splitLine :: String -> [Text]
splitLine line = Text.splitOn (Text.pack ",") (Text.pack line)

-- gets just the element's point, doing the right conversions
getDoubles :: Read a => [Text] -> [a]
getDoubles line = [read (Text.unpack x) | x <- init line]

-- gets just the element's class
getClass :: String -> String
getClass line = Text.unpack (last (splitLine line))

-- wraps all the previous functions, converting a string to useful data
lineToData :: String -> ([Double],String)
lineToData line = makeTuple (getDoubles (splitLine line)) (getClass line)

-- creates a tuple
makeTuple :: a -> b -> (a, b)
makeTuple x y = (x, y)

-- simple output reading function
readOutput :: IO String
readOutput = do
    putStrLn "Forneca o nome do arquivo de saida:"
    getLine

-- simple seed reading function
readSeed :: IO String
readSeed = do
    putStrLn "Forneca o valor da semente para geracao randomizada:"
    getLine

-- simple seed reading function
readFolds :: IO String
readFolds = do
    putStrLn "Forneca o número de folds:"
    getLine

-- simple seed reading function
readNeighbours :: IO String
readNeighbours = do
    putStrLn "Forneca o número de vizinhos:"
    getLine

-- teacher's function to remove duplicate values from a list
removeDup :: Eq a => [a] -> [a]
removeDup l = removeD l []
   where
     removeD [] _ = []
     removeD (x:xs) ls
        | x `elem` ls = removeD xs ls
        | otherwise = x: removeD xs (x:ls)

-- generates a list of random numbers given a seed, a size and a limit to the values
genRandom :: Int -> Int -> Int -> [Int]
genRandom seed testSize limit = take testSize (removeDup (randomRs (0,limit-1) $ mkStdGen seed:: [Int]))

getWholeFoldGroups :: [Int] -> Int -> Int ->[[Int]]
getWholeFoldGroups _ _ 0 = []
getWholeFoldGroups randomList foldSize n = take foldSize randomList : getWholeFoldGroups (drop foldSize randomList) foldSize (n-1)

insertRemaining :: [[Int]] -> [Int] -> [[Int]]
insertRemaining wholeFoldList remainings = [wholeFoldList !! x ++ [remainings !! x] | x <- [0..length remainings - 1]] ++ drop (length remainings) wholeFoldList

getFoldsIndex :: [Int] -> Int -> [[Int]]
getFoldsIndex randomList nFolds = insertRemaining (getWholeFoldGroups randomList (length randomList `div` nFolds) nFolds) (drop (nFolds * (length randomList `div` nFolds)) randomList)

--calculates a distance between two points
calcDist :: ([Double], String) -> ([Double], String) -> Double
calcDist (xs, _) (ys, _) = sqrt . sum $ [uncurry (-) z ** 2 | z <- zip xs ys]

-- function that tries to guess a point's class with a given training group
guessClassPoint :: [([Double], String)] -> ([Double], String) -> ([Double], String)
guessClassPoint trainingGroup point = makeTuple (fst point) (snd (trainingGroup !! Maybe.fromMaybe (-1) (elemIndex (minimum list) list))) 
    where
        list = [calcDist x point | x <- trainingGroup]

-- classifies all points of a given test group
classifyAllItems :: [([Double], String)] -> [([Double], String)] -> [([Double], String)]
classifyAllItems trainingGroup testGroup = [ guessClassPoint trainingGroup x | x <- testGroup]

-- classifies all points using cross validation
classifyAllItemsCV :: [([Double], String)] -> [[Int]] -> [([Double], String)]
classifyAllItemsCV dataList foldIndex = concat [classifyAllItems (getTrainingList dataList x) (getTestList dataList x) | x <- foldIndex]

classifyAllCentroidsCV :: [([Double], String)] -> [[Int]] -> [([Double], String)]
classifyAllCentroidsCV dataList foldIndex = concat [classifyAllItems (calcAllCentroids (getTrainingList dataList x)) (getTestList dataList x) | x <- foldIndex]

-- gets the test list
getTestList :: [([Double], String)] -> [Int] -> [([Double], String)]
getTestList dataList testList = [dataList !! x | x <- testList]

-- gets the training list
getTrainingList :: [([Double], String)] -> [Int] -> [([Double], String)]
getTrainingList dataList testList = [dataList !! x | x <- [0..length dataList - 1], x `notElem` testList]

-- counts the number off total right guesses the program did
rightCount :: [([Double], String)] -> [([Double], String)] -> Int -> Int
rightCount [] [] value = value
rightCount (x:xs) (y:ys) value =
    if snd x == snd y then
        rightCount xs ys (succ value)
        else rightCount xs ys value

-- calculates the accuracy of the program
calcAccuracy :: Int -> Int -> Double
calcAccuracy right total = fromIntegral (100 * right) / fromIntegral total

accuracyOneNeighbour :: [([Double], String)] -> [[Int]] -> Int -> Double
accuracyOneNeighbour dataList foldIndex nFolds = sum [calcAccuracy (rightCount (getTestList dataList x) (classifyAllItems (getTrainingList dataList x) (getTestList dataList x)) 0) (length x) | x <- foldIndex] / fromIntegral nFolds

-- gets all elements of a given class
getAllElementsInClass :: [([Double], String)] -> String -> [[Double]]
getAllElementsInClass dataList className = [ fst x | x <- dataList, snd x == className]

-- calculates a centroid of given points
calcCentroid :: [[Double]] -> [Double]
calcCentroid [] = []
calcCentroid [element] = element
calcCentroid (element : elements) = zipWith (+) element (calcCentroid elements)

-- get a list of all classes in a given list
getClassList :: Ord a1 => [(a2, a1)] -> [a1]
getClassList dataList = removeDup [snd x | x <- dataList]

--calculates all centroids
calcAllCentroids :: [([Double], String)] -> [([Double], String)]
calcAllCentroids dataList = [makeTuple (map (/ fromIntegral (length (getAllElementsInClass dataList x))) (calcCentroid (getAllElementsInClass dataList x))) x | x <- getClassList dataList]

accuracyCentroids :: [([Double], String)] -> [[Int]] -> Int -> Double
accuracyCentroids dataList foldIndex nFolds = sum [calcAccuracy (rightCount (getTestList dataList x) (classifyAllItems (calcAllCentroids (getTrainingList dataList x)) (getTestList dataList x)) 0) (length x) | x <- foldIndex] / fromIntegral nFolds

-- informs the number of times the program guessed a class "guessedClass" and the right class was "rightClass", usefull with the function below
calcGuesses :: [([Double], String)] -> [([Double], String)] -> String -> String -> Int -> Int
calcGuesses [] [] _ _ value = value
calcGuesses  (x:xs) (y:ys) guessedClass rightClass value =
    if snd x == guessedClass && snd y == rightClass then
        calcGuesses xs ys guessedClass rightClass (succ value)
        else calcGuesses xs ys guessedClass rightClass value

-- calculates the whole confusion table (aka confusion matrix)
confusionTable :: [([Double], String)] -> [([Double], String)] -> [String] -> [[Int]]
confusionTable guessedGroup testGroup classes = [[ calcGuesses guessedGroup testGroup x y 0 | y <- classes ] | x <- classes]

-- right align a number, and transforms it into a string
fixNumberPosition :: Int -> String
fixNumberPosition x
    | x > 9    = show x
    | otherwise = " " ++ show x

-- creates a line to be shown in the confusion table
createConfTableLine :: [String] -> [Char] -> [Char]
createConfTableLine [] line = line
createConfTableLine [x] _ = x
createConfTableLine (x:xs) line = line ++ x ++ ", " ++ createConfTableLine xs line

-- displays the confusion table
printConfTable :: [[Int]] -> String
printConfTable confTable = unlines [" " ++ createConfTableLine [fixNumberPosition x | x <- y] "" | y <- confTable]