package intermidia;

import java.io.FileReader;
import java.io.FileWriter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.openimaj.data.DataSource;
import org.openimaj.feature.DoubleFVComparison;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.feature.local.quantised.QuantisedLocalFeature;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.keypoints.Keypoint;
import org.openimaj.image.feature.local.keypoints.KeypointLocation;
import org.openimaj.math.statistics.distribution.Histogram;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.pair.IntFloatPair;

import com.opencsv.CSVReader;

import TVSSUnits.Shot;
import TVSSUnits.ShotList;

public class BoVWCalculator 
{
	
	private static int k = 500;
	private static int clusteringSteps = 50;
	
    public static void main( String[] args ) throws Exception 
    {    	
    	//Read SIFT features from CSV file.
    	CSVReader featureReader = new CSVReader(new FileReader(args[0]), ' ');
		String [] line;
		ShotList shotList = new ShotList();
		int lastShot = -1;
		
		
		
		//Build shot list with SIFT keypoints
		while ((line = featureReader.readNext()) != null) 
		{
			int currentShot = Byte.parseByte(line[0]);
			//It must be a while because there can be shots without keypoints
			while(currentShot != lastShot)
			{
				shotList.addShot(new Shot());
				lastShot++;
			}
			
			int fvSize = line.length - 1;
			byte fv[] = new byte[fvSize];
			
			for(int i = 0; i < fvSize; i++)
			{
				fv[i] = Byte.parseByte(line[i + 1]);
			}
			shotList.getLastShot().addSiftKeypoint(new Keypoint(0, 0, 0, 0, fv));
		}
		featureReader.close();
		
		//Build SIFT map per shot
		Map<Shot, LocalFeatureList<Keypoint>> videoKeypoints = new HashMap<Shot, LocalFeatureList<Keypoint>>();
		for(Shot shot: shotList.getList())
		{
			videoKeypoints.put(shot, shot.getSiftKeypointList());			
		}
		
		//Compute feature dictionary
		DataSource<byte []> kmeansDataSource = new LocalFeatureListDataSource<Keypoint, byte[]>(videoKeypoints);
		ByteKMeans clusterer = ByteKMeans.createExact(k, clusteringSteps);
		//$centroids have size $k, and each vector have 128 bytes
		System.out.println("Clustering SIFT Keypoints into "+ k + " visual words.");
		ByteCentroidsResult centroids = clusterer.cluster(kmeansDataSource);
		
		
		//Create the assigner, it is capable of assigning a feature vector to a cluster (to a centroid)
		HardAssigner<byte[], float[], IntFloatPair> hardAssigner = centroids.defaultHardAssigner();
		
		
    	//Compute features of each shot
		int shotn = 0;
		FileWriter bovwWriter = new FileWriter(args[1]);
		for(Shot shot: shotList.getList())
		{
			System.out.println("Processing shot " + shotn);
			//Print shot number
			bovwWriter.write(Integer.toString(shotn++));
			
			//Variable quantisedFeatures assign a cluster label between [1..k] to each feature vector from the list 
			List<QuantisedLocalFeature<KeypointLocation>> quantisedFeatures = BagOfVisualWords.computeQuantisedFeatures(hardAssigner, shot.getSiftKeypointList());

			//Create the visual word ocurrence histogram
			SparseIntFV features = BagOfVisualWords.extractFeatureFromQuantised(quantisedFeatures, k);
			
			//Set shot feature histogram for use in intershot distance
			Histogram featureHistogram = new Histogram(features.asDoubleVector());
			featureHistogram = new Histogram(featureHistogram.normaliseFV());
			shot.setFeatureWordHistogram(featureHistogram);
			
			
			for(int i = 0; i < features.length(); i++)
			{
				bovwWriter.write(" " + features.getVector().get(i));
			}
			bovwWriter.write("\n");			
		}
		bovwWriter.close();
		
		//Print visual words to file
		FileWriter vwWriter = new FileWriter(args[2]);
		for(int i = 0; i < centroids.numClusters(); i++)
		{
			for(int j = 0; j < centroids.numDimensions(); j++)
			{
				if(j < centroids.numDimensions() - 1)
				{
					vwWriter.write(centroids.getCentroids()[i][j] + " ");
				}else
				{
					vwWriter.write(centroids.getCentroids()[i][j] + "\n");
				}
			}
		}
		vwWriter.close();
		
		//Print intershot distances
		for(int i = 0; i < (shotList.listSize() - 1); i++)
		{
			double intershotDist = shotList.getShot(i).getFeatureWordHistogram().compare(shotList.getShot(i + 1).getFeatureWordHistogram(), 
					DoubleFVComparison.COSINE_SIM);
			System.out.println("Sim " +  i + "/" + (i + 1) + ": " + intershotDist);
		}
    }
}

