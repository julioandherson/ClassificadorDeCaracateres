package classificador;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;

import javax.imageio.ImageIO;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.ClassificationViaClustering;
import weka.classifiers.meta.RandomSubSpace;
import weka.classifiers.rules.NNge;
import weka.classifiers.rules.PART;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class ImagesSetup {

    private static final String DIGITOS = "digitos";
    private static final String LETRAS = "letras";
    private static final String DIGITOS_LETRAS = "digitos_letras";
    private static final String SEM_CARACTERES = "sem_caracteres";
    private static final int CAPACITY = 257;
    private static final int INDEX = 256;

    public static void main(String[] args) throws Exception {
        boolean verbose = Boolean.parseBoolean(args[0]);
        String tecnica = args[1];
        String treinamento = args[2];
        String teste = args[3];
        Classifier classificador;
        
        switch (Integer.parseInt(tecnica)) {
		case 0:
			classificador = new RandomSubSpace();
			break;
        case 1:
			classificador = new PART();
			break;
		case 2:
			classificador = new ClassificationViaClustering();
			break;
		case 3:
			classificador = new NNge();
			break;
		default:
			classificador = new RandomSubSpace();
			break;
		}

        FastVector wekaAttributes = new FastVector(CAPACITY);
        for (int i = 0; i < INDEX; i++) {
            Attribute attr = new Attribute("numeric" + i);
            wekaAttributes.addElement(attr);
        }
        FastVector classes = new FastVector(4);
        classes.addElement(DIGITOS);
        classes.addElement(LETRAS);
        classes.addElement(DIGITOS_LETRAS);
        classes.addElement(SEM_CARACTERES);
        Attribute attr = new Attribute("classes", classes);

        wekaAttributes.addElement(attr);
        Instances isTrainingSet = new Instances("Rel", wekaAttributes, 1);
        isTrainingSet.setClassIndex(INDEX);
       
        String folderDigits = treinamento + "/digitos";
        String folderLetters = treinamento + "/letras";
        String folderBoth = treinamento + "/digitos_letras";
        String nothing = treinamento + "/sem_caracteres";
        buildTrainingSet(wekaAttributes, isTrainingSet, folderDigits, DIGITOS);
        buildTrainingSet(wekaAttributes, isTrainingSet, folderLetters, LETRAS);
        buildTrainingSet(wekaAttributes, isTrainingSet, folderBoth, DIGITOS_LETRAS);
        buildTrainingSet(wekaAttributes, isTrainingSet, nothing, SEM_CARACTERES);
        classificador.buildClassifier(isTrainingSet);

        Evaluation eTest = new Evaluation(isTrainingSet);
        Instances testingSet = new Instances("Reltst", wekaAttributes, 1);
        testingSet.setClassIndex(INDEX);
        String folderTestLetters = teste + "/letras";
        String folderTestDigits = teste + "/digitos";
        String folderTestBoth = teste + "/digitos_letras";
        String nothingTest = teste + "/sem_caracteres";
        buildTrainingSet(wekaAttributes, testingSet, folderTestLetters, LETRAS);
        buildTrainingSet(wekaAttributes, testingSet, folderTestDigits, LETRAS);
        buildTrainingSet(wekaAttributes, testingSet, folderTestBoth, DIGITOS_LETRAS);
        buildTrainingSet(wekaAttributes, testingSet, nothingTest, SEM_CARACTERES);
        eTest.evaluateModel(classificador, testingSet);

        if(verbose) {
            System.out.println(eTest.toSummaryString(true));
            System.out.println(eTest.toClassDetailsString());
        }
        System.out.println("Precision: " + eTest.weightedPrecision());
        System.out.println("Recall: " + eTest.weightedRecall());
        System.out.println("F-Measure: " + eTest.weightedFMeasure());
    }

    private static void buildTrainingSet(FastVector wekaAttributes, Instances isTrainingSet, String folderName, String classe) throws Exception {
        File folder = new File(folderName);
        File[] listOfFiles = folder.listFiles();
        for (File f : listOfFiles) {
            double[] histogram = buildHistogram(f);
            createTrainingSet(isTrainingSet, wekaAttributes, histogram, classe);
        }
    }

    private static void createTrainingSet(Instances isTrainingSet, FastVector wekaAttributes, double[] histogram, String classe) {

        Instance imageInstance = new Instance(CAPACITY);
        for (int i = 0; i < histogram.length; i++) {
            imageInstance.setValue((Attribute) wekaAttributes.elementAt(i), histogram[i]);
        }
        if (!classe.isEmpty()) {
            imageInstance.setValue((Attribute) wekaAttributes.elementAt(INDEX), classe);
        }
        isTrainingSet.add(imageInstance);
    }

//////////////// helper code /////////////////////////

    private static final double LUMINANCE_RED = 0.299D;
    private static final double LUMINANCE_GREEN = 0.587D;
    private static final double LUMINANCE_BLUE = 0.114;
    private static final int HIST_WIDTH = 256;
    private static final int HIST_HEIGHT = 100;

    /**
     * Parses pixels out of an image file, converts the RGB values to
     * its equivalent grayscale value (0-255), then constructs a
     * histogram of the percentage of counts of grayscale values.
     *
     * @param file - the image file.
     * @return - a histogram of grayscale percentage counts.
     */
//    protected static double[] buildHistogram(File file) throws Exception {
//        BufferedImage image = ImageIO.read(file);
//        
//        int[][] pixelsMatrix = getPixelsMatrix(image);
//        int[][] grayscalePixelsMatrix = pixelsMatrix;
//
//        int width = pixelsMatrix[0].length;
//        int height = pixelsMatrix.length;
//        
//        List<Integer> graylevels = new ArrayList<Integer>();
//
//        for(int i = 0; i < width; i++){
//        	for (int j = 0; j < height; j++){
//                Color c = new Color(image.getRGB(i, j));
//                int graylevel = getGrayLevel(c);
//        		grayscalePixelsMatrix[j][i] = graylevel;
//                graylevels.add(graylevel);
//        	}
//        }
//        
//        double[] histogram = new double[HIST_WIDTH];
//        for (Integer graylevel : (new HashSet<Integer>(graylevels))) {
//            int idx = graylevel;
//            histogram[idx] +=
//                    Collections.frequency(graylevels, graylevel) * HIST_HEIGHT / height;
//        }
//        return histogram;
//    }
    
    
          protected static double[] buildHistogram(File file) throws Exception {
        BufferedImage image = ImageIO.read(file);
        
        int width = image.getWidth();
        int height = image.getHeight();
        
        List<Integer> graylevels = new ArrayList<Integer>();
        double maxWidth = 0.0D;
        double maxHeight = 0.0D;
        for (int row = 0; row < width; row++) {
            for (int col = 0; col < height; col++) {
                Color c = new Color(image.getRGB(row, col));
                int graylevel = getGrayLevel(c);
                graylevels.add(graylevel);
                maxHeight++;
                if (graylevel > maxWidth) {
                    maxWidth = graylevel;
                }
            }
        }
        
        double[] histogram = new double[HIST_WIDTH];
        for (Integer graylevel : (new HashSet<Integer>(graylevels))) {
            int idx = graylevel;
            histogram[idx] +=
                    Collections.frequency(graylevels, graylevel) * HIST_HEIGHT / maxHeight;
        }
        return histogram;
    }
     
    
    
    private static int[][] getPixelsMatrix(BufferedImage image) {

        final byte[] pixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        final int width = image.getWidth();
        final int height = image.getHeight();
        final boolean hasAlphaChannel = image.getAlphaRaster() != null;

        int[][] result = new int[height][width];
        if (hasAlphaChannel) {
           final int pixelLength = 4;
           for (int pixel = 0, row = 0, col = 0; pixel < pixels.length; pixel += pixelLength) {
              int argb = 0;
              argb += (((int) pixels[pixel] & 0xff) << 24); // alpha
              argb += ((int) pixels[pixel + 1] & 0xff); // blue
              argb += (((int) pixels[pixel + 2] & 0xff) << 8); // green
              argb += (((int) pixels[pixel + 3] & 0xff) << 16); // red
              result[row][col] = argb;
              col++;
              if (col == width) {
                 col = 0;
                 row++;
              }
           }
        } else {
           final int pixelLength = 3;
           for (int pixel = 0, row = 0, col = 0; pixel < pixels.length; pixel += pixelLength) {
              int argb = 0;
              argb += -16777216; // 255 alpha
              argb += ((int) pixels[pixel] & 0xff); // blue
              argb += (((int) pixels[pixel + 1] & 0xff) << 8); // green
              argb += (((int) pixels[pixel + 2] & 0xff) << 16); // red
              result[row][col] = argb;
              col++;
              if (col == width) {
                 col = 0;
                 row++;
              }
           }
        }

        return result;
     }
    
    private static int getGrayLevel(Color color){
    	 return (int) (LUMINANCE_RED * color.getRed() + LUMINANCE_GREEN * color.getGreen() + LUMINANCE_BLUE * color.getBlue());
    }
}