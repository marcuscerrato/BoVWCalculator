package intermidia;

import java.io.FileReader;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.openimaj.data.DataSource;
import org.openimaj.feature.DoubleFV;
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
	private final static int k = 1000;
	private final static int clusteringSteps = 50;
	
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
		ByteCentroidsResult centroids = clusterer.cluster(kmeansDataSource);
		
		
		//Create the assigner, it is capable of assigning a feature vector to a cluster (to a centroid)
		HardAssigner<byte[], float[], IntFloatPair> hardAssigner = centroids.defaultHardAssigner();
		
		
    	//Compute features of each shot
		//int shotn = 0;
		double minVal = Double.MAX_VALUE;
		double maxVal = Double.MIN_VALUE;
		for(Shot shot: shotList.getList())
		{
			//Variable quantisedFeatures assign a cluster label between [1..k] to each feature vector from the list 
			List<QuantisedLocalFeature<KeypointLocation>> quantisedFeatures = BagOfVisualWords.computeQuantisedFeatures(hardAssigner, shot.getSiftKeypointList());

			//Create the visual word ocurrence histogram
			SparseIntFV features = BagOfVisualWords.extractFeatureFromQuantised(quantisedFeatures, k);
			
/*			System.out.println("FVlen: " + features.length());
			for(int i = 0; i < features.length(); i++)
				System.out.print(features.getVector().get(i) + " ");
			System.out.println();*/
			
			
			DoubleFV array = features.asDoubleFV();
			for(int i = 0; i < array.length(); i++)
			{
				if(minVal > array.get(i))
				{
					minVal = array.get(i);
				}
				if(maxVal < array.get(i))
						{
					maxVal = array.get(i);
				}
			}
/*			System.out.println("DoubleFVlen: " + array.length());
			for(int i = 0; i < array.length(); i++)
				System.out.print(array.get(i) + " ");
			System.out.println();
*/			
			
			Histogram histogram = new Histogram(array);
/*			System.out.println("Histogramlen: "+ histogram.length());
			for(int i = 0; i < histogram.length(); i++)
				System.out.print(histogram.get(i) + " ");
			System.out.println();*/

			shot.setVisualWordHistogram(histogram);
			
/*			System.out.println("NormHistLen: "+ histogram.length());
			for(int i = 0; i < histogram.length(); i++)
				System.out.print(String.format("%.2f", histogram.get(i)) + " ");
			System.out.println();
			
			System.exit(0);
*/			

			
/*			if(shotn < 10)
				System.out.print("Shot  " + shotn++ + ":\t");
			else
				System.out.print("Shot  " + shotn++ + ":\t");			
			for(int i = 0; i < array.length; i++)
			{
				System.out.print("\t" + array.get(i));
			}
			System.out.println();
*/
		}
		
		//for(Shot shot: shotList.getList())
		//{
			//shot.setVisualWordHistogram(new Histogram(shot.getVisualWordHistogram().normaliseFV(minVal, maxVal)));
			//shot.setVisualWordHistogram(new Histogram(shot.getVisualWordHistogram().normaliseFV()));
		//}
		
		/*for(int i = 0; i < shotList.listSize(); i++)
		{
			System.out.print("Shot"+ i + " ");
			for(int j = 0; j < shotList.listSize(); j++)
			{
				Histogram histI = shotList.getShot(i).getVisualWordHistogram();
				Histogram histJ = shotList.getShot(j).getVisualWordHistogram();
				//System.out.print(String.format("%.2f", histI.compare(histJ, DoubleFVComparison.EUCLIDEAN)) + " ");
				//Cosine sim apparently shows better relationship than euclidean
				double sim = histI.compare(histJ, DoubleFVComparison.COSINE_SIM);
				if(Double.isNaN(sim))
				{
					System.out.print(String.format("%.2f", .0) + " ");
				}else
				{
					System.out.print(String.format("%.2f", sim) + " ");					
				}
			}
			System.out.println();
		}*/
    	
    	
    	

    	
    }
}

