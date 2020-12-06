import Text.Printf (printf)
import IOHandler
import Algorithms

main :: IO ()
main = do
    -- input/output/folds/neighbours/reading
    dataList <- readInput
    outputFileName <- readOutput
    folds <- readFolds
    knn <- readNeighbours
    seed <- readSeed

    -- generating random numbers and getting the 
    let randomList = genRandom (read seed) (length dataList) (length dataList)
    let foldIndex = getFoldsIndex randomList (read folds)

    -- classify the test points with one neighbour, by centroids and with k nearest neighbours (with all methods using cross validation)
    let oneNeighbour = classifyAllItemsCV dataList foldIndex
    let centroids = classifyAllCentroidsCV dataList foldIndex
    let kNeighbours = classifyAllKNNCV dataList foldIndex (read knn)

    -- prints every accuracy and std deviation 
    printf "Acuracia(vizinho): %.2f%%\n" $ accWZScore dataList oneNeighbour foldIndex (read folds)
    printf "Desvio-Padrao(vizinho): %.2f%%\n" $ stdDev dataList oneNeighbour foldIndex (read folds)
    printf "Acuracia(centroide): %.2f%%\n" $ accWZScore dataList centroids foldIndex (read folds)
    printf "Desvio-Padrao(centroide): %.2f%%\n" $ stdDev dataList centroids foldIndex (read folds)
    printf "Acuracia(k-vizinhos): %.2f%%\n" $ accWZScore dataList kNeighbours foldIndex (read folds)
    printf "Desvio-Padrao(k-vizinhos): %.2f%%\n" $ stdDev dataList kNeighbours foldIndex (read folds)

    -- evaluates each confusion table
    let confTableNeighbour = confusionTable (concat oneNeighbour) (getTestList dataList (concat foldIndex)) (getClassList dataList)
    let confTableCentroid = confusionTable (concat centroids) (getTestList dataList (concat foldIndex)) (getClassList dataList)
    let confTableKN = confusionTable (concat kNeighbours) (getTestList dataList (concat foldIndex)) (getClassList dataList)

    -- prints confusion tables to output file
    genOutputFile outputFileName confTableNeighbour confTableCentroid confTableKN
    
    -- all done :) end of main
