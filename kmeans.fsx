open System

module Utils =

    let readDataPointsFile() =
        System.IO.File.ReadLines(@"\test.data")
        |> Seq.skip 1
        |> Seq.map (fun s -> s.Split(' ') |> Array.map float)       

    let readGroundTruthFile() =
        System.IO.File.ReadLines(@"\test.ground")
        |> Seq.skip 1
        |> Seq.map int

    let dataPoints = readDataPointsFile()
    let groundTruth = readGroundTruthFile()

    let squaredDistance a b = 
        Array.fold2 (fun acc p1 p2 -> acc + (p1-p2)**2.) 0. a b
        
module KMeans =

    let euclideanDistance a b = 
        Array.fold2 (fun acc p1 p2 -> acc + (p1-p2)**2.) 0. a b
        |> sqrt
    
    let computeSSE (data: float[][]) (centers: float[][]) (clusterID: int array) =
        let mutable sse = 0.
        data
        |> Array.iteri (fun i d ->
            let c = centers.[clusterID.[i]]
            sse <- sse + Utils.squaredDistance d c)
        sse

    // K-means clustering using k clusters and at most maxIter iterations or until tolerance tol is met.
    // Parameter distanceFun is a distance function (e.g., euclidean distance, Manhattan distance)
    let kmeans (data: float[][]) k maxIter tol distanceFun =
        let dimensions = Array.length data.[0]

        let rec update (centers: float[][]) (clusterID: int array) iter lastDistance =
            if iter = 0 then Array.toList clusterID
            else 
                let csums = Array2D.zeroCreate<float> k (dimensions+1)
                data |> Array.iteri (fun di da -> 
                    let thisClusterID = centers |> Array.mapi (fun ci ca -> (ci, distanceFun da ca)) |> Seq.minBy (fun (_, d) -> d) |> fst
                    clusterID.[di] <- thisClusterID      // This data point with index di belongs now to cluster thisClusterID
                    da |> Array.iteri (fun i p -> csums.[thisClusterID,i] <- csums.[thisClusterID,i] + p)    // Sum of this cluster's coordinate values
                    csums.[thisClusterID,dimensions] <- csums.[thisClusterID,dimensions] + 1.)    // Number of data points in this cluster
                for ci = 0 to k-1 do        // Recalculate all cluster centers by taking the mean of its data points
                    for j = 0 to dimensions-1 do
                        centers.[ci].[j] <- if csums.[ci,dimensions] > 0. then csums.[ci,j]/csums.[ci,dimensions] else 0.
                let distance = computeSSE data centers clusterID
                let mutable itersLeft = iter-1
                printfn "# iterations: %d\nSSE: %f" (maxIter-itersLeft) distance
                if lastDistance - distance < tol || (lastDistance - distance)/lastDistance < tol then itersLeft <- 0
                update centers clusterID itersLeft distance

        let centers = [| for i=0 to k-1 do 
                            let newpoint = Array.zeroCreate<float> dimensions
                            for j=0 to dimensions-1 do
                                newpoint.[j] <- data.[i].[j]
                            yield newpoint |]    // Take the first k data points as initial cluster centers, make deep copy
        let clusterID = Array.zeroCreate<int> data.Length   // Holds the cluster assignment of each data point
        update centers clusterID maxIter System.Double.PositiveInfinity

module SupervisedEvaluation =
    // Purity: Quantifies the extent that each cluster contains only points from one ground truth partition.
    // The closer to 1 the purer.
    let purity groundtruthAssignment algorithmAssignment =
        let purcounts = List.zip algorithmAssignment groundtruthAssignment 
                        |> List.groupBy fst     // Group by assigned clusters
                        |> List.fold (fun acc (_,c) -> acc + (c |> List.countBy (fun (_,co) -> co) |> List.maxBy snd |> snd)) 0
        float purcounts / float algorithmAssignment.Length        // Divide by total number of data points

    // Normalized Mutual Information (NMI): Quantifies the amount of shared information between the clustering and 
    // ground truth partitioning. Value range [0,1]. Value close to 1 indicates a good clustering.
    let NMI groundtruthAssignment algorithmAssignment =
        let r = algorithmAssignment |> List.distinct |> List.length
        let k = groundtruthAssignment |> List.distinct |> List.length
        let n = float algorithmAssignment.Length
        let p = Array2D.zeroCreate<float> r k
        let pc = Array.zeroCreate<float> r
        let pt = Array.zeroCreate<float> k
        List.zip algorithmAssignment groundtruthAssignment 
        |> List.groupBy fst
        |> List.iter (fun (ci,c) ->
            c 
            |> List.countBy (fun (_,j) -> j)
            |> List.iter (fun (j,count) ->
                let prob = (float count)/n
                p.[ci, j] <- prob
                pc.[ci] <- pc.[ci] + prob
                pt.[j] <- pt.[j] + prob))
        let hc = Array.sumBy (fun x -> -x * System.Math.Log(x)) pc
        let ht = Array.sumBy (fun x -> -x * System.Math.Log(x)) pt
        let mutable ict = 0.
        for i = 0 to r-1 do
            for j = 0 to k-1 do
                let v = p.[i,j]/(pc.[i]*pt.[j])
                if v > 0. then ict <- ict + p.[i,j] * System.Math.Log(v)
        ict / sqrt (hc*ht)

module KernelKMeans =
    
    // Transform the data points to kernel space using Gaussian radial basis function (RBF).
    let kernel (data: float[][]) sigma =
        let sigma2sq = 2.*sigma**2.
        let n = Array.length data
        let km = [| for i = 0 to n-1 do yield Array.create n 1. |]
        for i = 0 to n-1 do
            for j = 0 to n-1 do
                if j>i then km.[i].[j] <- Math.Exp(- (Utils.squaredDistance data.[i] data.[j])/sigma2sq)
                elif j<i then km.[i].[j] <- km.[j].[i]
        km

// Main
let data = Utils.readDataPointsFile() |> Seq.toArray
let groundtruth = Utils.readGroundTruthFile() |> Seq.toList
// If Kernel K-means then uncomment the following line to transform input data and input dataTrans 
// to the KMeans.kmeans function instead of data:
// let dataTrans = KernelKMeans.kernel data 4.
let result = KMeans.kmeans data 2 100 1e-6 KMeans.euclideanDistance
printfn "Purity = %f" (SupervisedEvaluation.purity groundtruth result)
printfn "NMI = %f" (SupervisedEvaluation.NMI groundtruth result)
