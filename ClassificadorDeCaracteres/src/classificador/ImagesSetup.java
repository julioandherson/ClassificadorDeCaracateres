package classificador;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;

import javax.imageio.ImageIO;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.meta.ClassificationViaClustering;
import weka.classifiers.meta.RandomSubSpace;
import weka.classifiers.rules.NNge;
import weka.classifiers.rules.PART;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class ImagesSetup {
	// true 0 imagens/treinamento imagens/teste
	private static final String DIGITOS = "digitos";
	private static final String LETRAS = "letras";
	private static final String DIGITOS_LETRAS = "digitos_letras";
	private static final String SEM_CARACTERES = "sem_caracteres";
	private static final int CAPACITY = 769;
	private static final int INDEX = 768;

	public static void main(String[] args) throws Exception {
		boolean verbose = Boolean.parseBoolean(args[0]);
		String tecnica = args[1];
		String treinamento = args[2];
		String teste = args[3];
		Classifier classificador;

		switch (Integer.parseInt(tecnica)) {
		case 0:
			classificador = new BayesNet();
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
		Instances isTrainingSet = new Instances("Rel", wekaAttributes, 100);
		isTrainingSet.setClassIndex(INDEX);

		String folderDigits = treinamento + "/digitos";
		String folderLetters = treinamento + "/letras";
		String folderBoth = treinamento + "/digitos_letras";
		String nothing = treinamento + "/sem_caracteres";
		buildTrainingSet(wekaAttributes, isTrainingSet, folderDigits, DIGITOS);
		buildTrainingSet(wekaAttributes, isTrainingSet, folderLetters, LETRAS);
		buildTrainingSet(wekaAttributes, isTrainingSet, folderBoth,
				DIGITOS_LETRAS);
		buildTrainingSet(wekaAttributes, isTrainingSet, nothing, SEM_CARACTERES);
		classificador.buildClassifier(isTrainingSet);

		Evaluation eTest = new Evaluation(isTrainingSet);
		Instances testingSet = new Instances("Reltst", wekaAttributes, 100);
		testingSet.setClassIndex(INDEX);
		String folderTestLetters = teste + "/letras";
		String folderTestDigits = teste + "/digitos";
		String folderTestBoth = teste + "/digitos_letras";
		String nothingTest = teste + "/sem_caracteres";
		buildTrainingSet(wekaAttributes, testingSet, folderTestDigits, DIGITOS);
		buildTrainingSet(wekaAttributes, testingSet, folderTestLetters, LETRAS);
		buildTrainingSet(wekaAttributes, testingSet, folderTestBoth,
				DIGITOS_LETRAS);
		buildTrainingSet(wekaAttributes, testingSet, nothingTest,
				SEM_CARACTERES);
		eTest.evaluateModel(classificador, testingSet);

		if (verbose) {
			System.out.println(eTest.toSummaryString(true));
			System.out.println(eTest.toClassDetailsString());
		}
		System.out.println("Precision: " + eTest.weightedPrecision());
		System.out.println("Recall: " + eTest.weightedRecall());
		System.out.println("F-Measure: " + eTest.weightedFMeasure());
	}

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

	private static final int HIST_WIDTH = 768;
	private static final int HIST_HEIGHT = 100;

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
				graylevels.add(c.getRed());
				graylevels.add(c.getGreen());
				graylevels.add(c.getBlue());
				maxHeight++;
				if (c.getRed() > maxWidth) {
					maxWidth = c.getRed();
				}else if (c.getGreen() > maxWidth) {
					maxWidth = c.getGreen();
				}else if(c.getBlue() > maxWidth){
					maxWidth = c.getBlue();
				}
			}
		}

		double[] histogram = new double[HIST_WIDTH];
		for (Integer graylevel : (new HashSet<Integer>(graylevels))) {
			int idx = graylevel;
			histogram[idx] += Collections.frequency(graylevels, graylevel)
					* HIST_HEIGHT / maxHeight;
		}
		return histogram;
	}
}