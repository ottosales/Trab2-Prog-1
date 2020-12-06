module IOHandler(
    readInput,
    readOutput,
    readFolds,
    readNeighbours,
    readSeed,
    genOutputFile
)
where

import Data.Text as Text (Text, pack, unpack, splitOn)
import Algorithms ( printConfTable, makeTuple )
type DataTuple = ([Double], String)

-- | reads the input and returns a list of tuples, with the fst element of the tuple being the point (a list of Double) and the second a string (point's class)
readInput :: IO [DataTuple]
readInput = do
    putStrLn "Forneca o nome do arquivo de entrada:"
    fileName <- getLine
    handle <- readFile fileName
    return [lineToData handle | handle <- lines handle]

-- | splits the line at the ","
splitLine :: String -> [Text]
splitLine line = Text.splitOn (Text.pack ",") (Text.pack line)

-- | gets just the element's point, doing the right conversions
getDoubles :: Read a => [Text] -> [a]
getDoubles line = [read (Text.unpack x) | x <- init line]

-- | gets just the element's class
getClass :: String -> String
getClass line = Text.unpack (last (splitLine line))

-- | wraps all the previous functions, converting a string to useful data
lineToData :: String -> ([Double],String)
lineToData line = makeTuple (getDoubles (splitLine line)) (getClass line)

-- | simple output reading function
readOutput :: IO String
readOutput = do
    putStrLn "Forneca o nome do arquivo de saida:"
    getLine

-- | simple seed reading function
readSeed :: IO String
readSeed = do
    putStrLn "Forneca o valor da semente para geracao randomizada:"
    getLine

-- | simple seed reading function
readFolds :: IO String
readFolds = do
    putStrLn "Forneca o número de folds:"
    getLine

-- | simple seed reading function
readNeighbours :: IO String
readNeighbours = do
    putStrLn "Forneca o número de vizinhos:"
    getLine

-- | generates the output file with a sequence of confusion tables
genOutputFile :: FilePath -> [[Int]] -> [[Int]] -> [[Int]] -> IO ()
genOutputFile outputFileName confTableNeighbour confTableCentroid confTableKN = do
    writeFile outputFileName "vizinho mais proximo:\n"
    appendFile outputFileName $ printConfTable confTableNeighbour
    appendFile outputFileName "\ncentroides:\n"
    appendFile outputFileName $ printConfTable confTableCentroid
    appendFile outputFileName "\nk-vizinhos mais proximos:\n"
    appendFile outputFileName $ printConfTable confTableKN