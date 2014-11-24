
package classificador;

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

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;

import javax.imageio.ImageIO;

// TODO: Auto-generated Javadoc
/**
 * The Class ClassificadorDeCaracteres.
 */
public class ClassificadorDeCaracteres {
    // -v randomsubspace imagens/treinamento imagens/teste
    // "" randomsubspace imagens/treinamento imagens/teste

    /** The Constant DIGITOS. */
    private static final String DIGITOS = "digitos";

    /** The Constant LETRAS. */
    private static final String LETRAS = "letras";

    /** The Constant DIGITOS_LETRAS. */
    private static final String DIGITOS_LETRAS = "digitos_letras";

    /** The Constant SEM_CARACTERES. */
    private static final String SEM_CARACTERES = "sem_caracteres";

    /** The Constant CAPACITY. */
    private static final int CAPACITY = 257;

    /** The Constant INDEX. */
    private static final int INDEX = 256;

    /** The Constant LUMINANCE_RED. */
    private static final double LUMINANCE_RED = 0.299D;

    /** The Constant LUMINANCE_GREEN. */
    private static final double LUMINANCE_GREEN = 0.587D;

    /** The Constant LUMINANCE_BLUE. */
    private static final double LUMINANCE_BLUE = 0.114;

    /** The Constant HIST_WIDTH. */
    private static final int HIST_WIDTH = 256;

    /** The Constant HIST_HEIGHT. */
    private static final int HIST_HEIGHT = 100;

    /**
     * The main method.
     * 
     * @param args the arguments
     * @throws Exception the exception
     */
    public static void main(String[] args) throws Exception {
        String verbose = args[0];
        String technique = args[1];
        String training_Set = args[2];
        String test_Set = args[3];
        Classifier classifier = null;

        if (technique.toLowerCase().equals("randomsubspace")) {
            classifier = new RandomSubSpace();
        } else if (technique.toLowerCase().equals("part")) {
            classifier = new PART();
        } else if (technique.toLowerCase().equals("classificationviaclustering")) {
            classifier = new ClassificationViaClustering();
        } else if (technique.toLowerCase().equals("nnge")) {
            classifier = new NNge();
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

        String folderDigits = training_Set + "/digitos";
        String folderLetters = training_Set + "/letras";
        String folderBoth = training_Set + "/digitos_letras";
        String nothing = training_Set + "/sem_caracteres";
        buildTrainingSet(wekaAttributes, isTrainingSet, folderDigits, DIGITOS);
        buildTrainingSet(wekaAttributes, isTrainingSet, folderLetters, LETRAS);
        buildTrainingSet(wekaAttributes, isTrainingSet, folderBoth,
                DIGITOS_LETRAS);
        buildTrainingSet(wekaAttributes, isTrainingSet, nothing, SEM_CARACTERES);
        classifier.buildClassifier(isTrainingSet);

        Evaluation eTest = new Evaluation(isTrainingSet);
        Instances testingSet = new Instances("Reltst", wekaAttributes, 1);
        testingSet.setClassIndex(INDEX);

        String folderTestLetters = test_Set + "/letras";
        String folderTestDigits = test_Set + "/digitos";
        String folderTestBoth = test_Set + "/digitos_letras";
        String nothingTest = test_Set + "/sem_caracteres";

        buildTrainingSet(wekaAttributes, testingSet, folderTestLetters, LETRAS);
        buildTrainingSet(wekaAttributes, testingSet, folderTestDigits, DIGITOS);
        buildTrainingSet(wekaAttributes, testingSet, folderTestBoth,
                DIGITOS_LETRAS);
        buildTrainingSet(wekaAttributes, testingSet, nothingTest,
                SEM_CARACTERES);
        eTest.evaluateModel(classifier, testingSet);

        PrintWriter writer = new PrintWriter("out.txt", "UTF-8");

        if (verbose.toLowerCase().equals("-v")) {
            List<String> listOfDirectorys = new ArrayList<String>();
            listOfDirectorys.add(folderTestLetters);
            listOfDirectorys.add(folderTestDigits);
            listOfDirectorys.add(folderTestBoth);
            listOfDirectorys.add(nothingTest);

            for (String directory : listOfDirectorys) {
                File folder = new File(directory);
                File[] listOfFiles = folder.listFiles();
                for (File f : listOfFiles) {
                    System.out.println(f.getName() + ": ");
                    writer.println(f.getName() + ": ");
                }
            }
        }
        writer.println("\nPrecision: " + eTest.weightedPrecision());
        writer.println("Recall: " + eTest.weightedRecall());
        writer.println("F-Measure: " + eTest.weightedFMeasure());

        System.out.println("Precision: " + eTest.weightedPrecision());
        System.out.println("Recall: " + eTest.weightedRecall());
        System.out.println("F-Measure: " + eTest.weightedFMeasure());
        writer.close();
    }

    /**
     * Builds the training set.
     * 
     * @param wekaAttributes the weka attributes
     * @param isTrainingSet the is training set
     * @param folderName the folder name
     * @param classe the classe
     * @throws Exception the exception
     */
    private static void buildTrainingSet(FastVector wekaAttributes,
            Instances isTrainingSet, String folderName, String classe)
            throws Exception {
        File folder = new File(folderName);
        File[] listOfFiles = folder.listFiles();
        for (File f : listOfFiles) {
            double[] histogram = buildHistogram(f);
            createTrainingSet(isTrainingSet, wekaAttributes, histogram, classe);
        }
    }

    /**
     * Creates the training set.
     * 
     * @param isTrainingSet the is training set
     * @param wekaAttributes the weka attributes
     * @param histogram the histogram
     * @param classe the classe
     */
    private static void createTrainingSet(Instances isTrainingSet,
            FastVector wekaAttributes, double[] histogram, String classe) {
        Instance imageInstance = new Instance(CAPACITY);
        for (int i = 0; i < histogram.length; i++) {
            imageInstance.setValue((Attribute) wekaAttributes.elementAt(i),
                    histogram[i]);
        }
        if (!classe.isEmpty()) {
            imageInstance.setValue((Attribute) wekaAttributes.elementAt(INDEX),
                    classe);
        }
        isTrainingSet.add(imageInstance);
    }

    /**
     * Builds the histogram.
     * 
     * @param file the file
     * @return the double[]
     * @throws Exception the exception
     */
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
        for (Integer graylevel : new HashSet<Integer>(graylevels)) {
            int idx = graylevel;
            histogram[idx] += Collections.frequency(graylevels, graylevel)
                    * HIST_HEIGHT / maxHeight;
        }
        return histogram;
    }

    /**
     * Gets the gray level.
     * 
     * @param color the color
     * @return the gray level
     */
    private static int getGrayLevel(Color color) {
        return (int) (LUMINANCE_RED * color.getRed() + LUMINANCE_GREEN
                * color.getGreen() + LUMINANCE_BLUE * color.getBlue());
    }
}
