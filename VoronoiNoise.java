/*
 * Voroni/Worley noise with optional addition of fractional/fractl Brownian movement (fBm)
 * 
 * Image files are made with the Worley noise and with and without the fBm nosie
 * 
 * A video file is made with an animation of the noise image with slight movement of the cell centers
 * in each frame.
 */

// AI comments below don't go with the code below but are of general help
// Voronoi noise, also known as Worley noise, is a procedural noise function often used in computer graphics and procedural generation to create organic-looking textures and terrains. It works by dividing space into cells based on the distance to a set of "seed" points. The value at any given point is typically determined by the distance to the closest seed point, or a combination of distances to multiple seed points.
// Here's a conceptual outline of how a Voronoi noise implementation in Java might work, along with key components you'd need:
// 1. Seed Point Generation:
// You need a way to generate a set of random seed points within a given area or grid. A Random object with a seed for reproducibility is a good choice.
// These seed points can be represented by Point2D or a custom Vector2D class if you're working in 2D, or Point3D/Vector3D for 3D.
// 2. Noise Calculation Function:
// This function takes a coordinate (e.g., x, y) as input and returns the Voronoi noise value at that point.
// Grid Optimization: To improve performance, especially for larger areas, you can divide the space into a grid. For any given input coordinate, you only need to consider the seed points in the current grid cell and its immediate neighbors (e.g., 3x3 cells in 2D). This avoids iterating through all seed points for every pixel.
// Distance Calculation: For each seed point within the relevant grid cells, calculate the distance from the input coordinate to that seed point. Euclidean distance is common, but other distance metrics can be used for different visual effects.
// Finding the Closest: Determine the minimum distance among all calculated distances. This minimum distance is often the basic Voronoi noise value.
// Optional Features:
// Feature Points: You can store additional data with each seed point (e.g., a random value) and use that in the final noise calculation, instead of just the distance.
// Multiple Features (F1, F2, etc.): Instead of just the closest seed, you can calculate the distance to the 2nd closest, 3rd closest, etc., and use combinations of these distances (e.g., F2 - F1) to create different patterns.
// 3. Example Code Structure (Conceptual):

// Note: The provided conceptual code is a simplified example. A full-fledged Voronoi noise implementation often involves more sophisticated grid management and potentially different distance functions or combinations of features to achieve various visual effects. Libraries like FastNoiseLite or JMonkeyEngine might offer pre-built noise functions, including Voronoi, if you are looking for a ready-to-use solution.
package App;

import java.util.Arrays;
import java.util.Random;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoWriter;

public class VoronoiNoise {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME); // Load the native OpenCV library
      }

    // select distance calculation method
    private static enum DistanceMethod {Euclidean, Manhattan, Exponential}
    private static final DistanceMethod distanceMethod = DistanceMethod.Exponential;
    // select which points are used for distance calculation
    private static enum ClosestPoint {f1, f2, f3, f2_1, f3_2}
    private static final ClosestPoint ClosestPointfn = ClosestPoint.f1;
    // select if gray scale black to white is flipped white to black
    private static final boolean invertImageGray = true;
    // select if fBm is to be added to images
    private static final boolean useFBM = true;
    // select size of image
    private static int imageWidth = 500;
    private static int imageHeight = 500;
    // select of of rando points to use as cell centers
    private static int numberPoints = 50;
    // select number of frames to create for video at 10 fps
    private static int videoLength = 30;
    // name of video file
    private static final String VideoName = "Voroni.mp4";

    // Configure fBm parameters
    private static int octaves = 8;
    private static double persistence = 0.5;
    private static double scale = 0.01;

    // end of user options
    private static Mat image = new Mat(imageHeight, imageWidth, CvType.CV_8UC1);
    private static Mat image2 = new Mat(image.rows(), image.cols(), image.type());
    private static byte[] gray = new byte[image.cols()];
    private static byte[] gray2 = new byte[image2.cols()];
    private static int[] randomPointsX = new int[numberPoints];
    private static int[] randomPointsY = new int[randomPointsX.length];
    private static Random random = new Random(123L);
    private static Mat allOnes = Mat.zeros(image.rows(), image.cols(), image.type()); // must be filled later for all ones
    private static CoherentNoiseGenerator noiseGenerator = new CoherentNoiseGenerator();
    private static final MatOfInt writeParams = new MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, 100); // pair-wise; param1, value1, ...
    private static final VideoWriter outputVideo= new VideoWriter();
    private static final Size S = new Size(image.cols(),image.rows());
    private static final int ex = VideoWriter.fourcc('A', 'V', 'C', '1');

    public static void main(String[] args)
    {
        System.out.println(VideoName + " open returned " + outputVideo.open(VideoName, ex, 10., S, false));
        if (!outputVideo.isOpened())
        {
            System.out.println("Could not open the output video for write: " + VideoName);
            System.exit(1);
        }

        allOnes.setTo(new Scalar(255)); // used for the XOR inversion of the image gray scale
        createRandomCellCenters();
        for (int frame = 1; frame <= videoLength; frame++) // run to video length
        {
            moveCellCentersRandomly();
            createNoisyImages();

            if (invertImageGray)
            {
                Core.bitwise_xor(image, allOnes, image);
                if (useFBM)
                {
                    Core.bitwise_xor(image2, allOnes, image2);
                }
            }
            // likely full range of gray has been used but normalize assures this for best gray rendition
            Core.normalize(image, image,0, 255, Core.NORM_MINMAX);

            Imgcodecs.imwrite("VoroniNoise.jpg", image, writeParams); // save camera image
            HighGui.imshow("Voroni Noise", image);
            outputVideo.write(image/*threeChannelMat*/);

            //Caution: if fBm image is displayed, it appears exactly over the Worley image and must be moved to reveal Worley
            if (useFBM)
            {
                Core.normalize(image2, image2,0, 255, Core.NORM_MINMAX);
                Imgcodecs.imwrite("voroniNoise+Brownian.jpg", image2, writeParams); // save camera image
                HighGui.imshow("Voroni Noise + fBm", image2);                
            }

            HighGui.waitKey(10);
        }
        outputVideo.release();
    }

    /**
     * Create 2-D points for cell centers
     */
    private static void createRandomCellCenters()
    {
        for (int i = 0; i < randomPointsX.length; i++)
        {
            randomPointsX[i] = (int)((double)image.cols()*random.nextDouble());
            randomPointsY[i] = (int)((double)image.rows()*random.nextDouble());
        }
    }

    /**
     * Move cell centers slightly
     */
    private static void moveCellCentersRandomly()
    {
        for (int i = 0; i < randomPointsX.length; i++)
        {
            randomPointsX[i] = (int)((double)randomPointsX[i] + 10.*Math.sin(random.nextDouble()*2.-1.));
            randomPointsY[i] = (int)((double)randomPointsY[i] + 10.*Math.sin(random.nextDouble()*2.-1.));
        }
    }

    /**
     * Create images with Vorono/Worley noise and optionally add
     * fractional/fractl Brownian movement
     */
    private static void createNoisyImages()
    {
        for (int i = 0; i < image.rows(); i++)
        {
            for (int j = 0; j < image.cols(); j++)
            {
                var WorleyNoise = VoronoiNoise.noise((double)j, (double)i);
                gray[j] = (byte)WorleyNoise; // Java double to byte conversion is tricky
                if (useFBM)
                    {
                        var fBmNoise = fBm(j, i);
                        gray2[j] = (byte)(WorleyNoise + 128.*fBmNoise);  // Map normalized fbm noise (e.g., -1 to 1) to a color shift (e.g., -128 to 127)
                    }
            }
            image.put(i, 0, gray);
            if (useFBM)
            {
                image2.put(i, 0, gray2);
            }
        }
    }

    /**
     * Compute fBm for this 2-D point
     * @param x image x axis point
     * @param y image y axis point
     * @return amount of fBm noise from -1 to 1
     */
private static double fBm(int x, int y)
    {
        double totalNoise = 0.0;
        double amplitude = 1.0;
        double frequency = 1.0;
        double maxAmplitude = 0.0;

        for (int i = 0; i < octaves; i++) {
            totalNoise += noiseGenerator.noise(x * frequency * scale, y * frequency * scale) * amplitude;
            maxAmplitude += amplitude;
            amplitude *= persistence;
            frequency *= 2;
        }

        // Normalize the noise value to be between -1 and 1
        totalNoise = totalNoise / maxAmplitude;

        return totalNoise;
    }

    /**
     * Compute Worley/Voroni noise at this 2-D point
     * Noise is the distance to a close cell center as selected by the user
     * @param x image x axis point
     * @param y image y axis point
     * @return amount of Worley/Voroni noise
     */
    public static double noise(double x, double y) {

        double[] minDistance = new double[3];
        Arrays.fill(minDistance, Double.MAX_VALUE);
        double distance = Double.MAX_VALUE;
        
        // compute distance from this point to all the cell centers
        // looking for the closest distance as selected by the user
        for (int i = 0; i < randomPointsX.length; i++)
        {
            double dx = x - randomPointsX[i];
            double dy = y - randomPointsY[i];
            distance = getDistance(dx, dy);
            // there can be unlimited variations on what distance to use
            // a few variations are implemented
            if (distance <= minDistance[0])
            {
                minDistance[2] = minDistance[1]; // distance to third closest point
                minDistance[1] = minDistance[0]; // distance to second closest point
                minDistance[0] = distance; // distance to (first) closest point
            }
            else
            if (distance <= minDistance[1])
            {
                minDistance[2] = minDistance[1];
                minDistance[1] = distance;
            }
            else
            if (distance < minDistance[2])
            {
                minDistance[2] = distance;
            }
        }

        switch(ClosestPointfn)
        {
            case f1:
                return minDistance[0];
            case f2:
                return minDistance[1];
            case f3:
                return minDistance[2];
            case f2_1:
                return minDistance[1] - minDistance[0];
            case f3_2:
                return minDistance[2] - minDistance[1];
            default:
                return minDistance[0];
        }
    }

    /**
     * Compute distance between the difference between two points
     * There are several variations on distance calculations and some are implemented here
     * @param xDist difference between points in the x axis
     * @param zDist difference between points in the y axis
     * @return distance between two points
     */
    private static double getDistance(double xDist, double zDist) {
        switch(distanceMethod) {
            case Euclidean:
                return Math.sqrt(xDist * xDist + zDist * zDist); // Euclidean distance
            case Manhattan:
                return Math.abs(xDist) + Math.abs(zDist); // Manhattan distance L1 norm
            case Exponential: // an unusual distance function whose origin is unknown
                return Math.pow(Math.E, Math.sqrt(xDist * xDist + zDist * zDist) / Math.sqrt(2.))/Math.E;
            default:
                return 1.0; // junk
        }
    }
}

/*
        int t1 = (int)1.E20;
        int t2 = (int)-1.E20;
        byte b1 = (byte)t1;
        byte b2 = (byte)t2;
        System.out.println(t1 + " " + b1 + " " + t2 + " " + b2); // 2147483647 -1 -2147483648 0

        for (int i = 0; i < image.rows(); i++)
        {
            for (int j = 0; j < image.cols(); j++)
            {
                var temp = Math.pow(Math.E, Math.sqrt((double)j * (double)j + (double)i * (double)i) / Math.sqrt(2.))/Math.E;
                gray[j] = (byte)temp;
                System.out.printf("%d, %d   %f  %d\n", j, i, temp, gray[j]);
            }
            image2.put(i, 0, gray);
        }

        Core.normalize(image2, image2,0, 255, Core.NORM_MINMAX);
        Imgcodecs.imwrite("ExponentialDistance.jpg", image2, writeParams); // save camera image
        HighGui.imshow("Exponential Distance", image2);                

        HighGui.waitKey(1000);
        System.out.println(image2.dump());
        System.out.println(Long.MAX_VALUE + "  " + Integer.MAX_VALUE);

        // System.out.println("OpenCV version " + Core.getVersionString() + "\n" + Core.getBuildInformation());

        String NAME = "Voroni.avi";
        int ex = -1; // lists all the codecs; needs opencv_videoio_ffmpeg430_64.dll accessible
        int ex = VideoWriter.fourcc('M', 'J', 'P', 'G');
        set(CV_CAP_PROP_FOURCC, CV_FOURCC('H', '2', '6', '5')); 
 */