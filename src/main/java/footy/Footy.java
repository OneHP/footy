package footy;

import org.apache.commons.io.FilenameUtils;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.earlystopping.EarlyStoppingConfiguration;
import org.deeplearning4j.earlystopping.EarlyStoppingModelSaver;
import org.deeplearning4j.earlystopping.EarlyStoppingResult;
import org.deeplearning4j.earlystopping.saver.LocalFileModelSaver;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.earlystopping.termination.MaxEpochsTerminationCondition;
import org.deeplearning4j.earlystopping.termination.MaxTimeIterationTerminationCondition;
import org.deeplearning4j.earlystopping.termination.ScoreImprovementEpochTerminationCondition;
import org.deeplearning4j.earlystopping.trainer.EarlyStoppingTrainer;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * @author Adam Gibson
 */
public class Footy {

    private static Logger log = LoggerFactory.getLogger(Footy.class);

    public static void main(String[] args) throws  Exception {

        //First: get the dataset using the record reader. CSVRecordReader handles loading/parsing
        int numLinesToSkip = 0;
        String delimiter = ",";
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip,delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("footy3.csv").getFile()));

        //Second: the RecordReaderDataSetIterator handles conversion to DataSet objects, ready for use in neural network
        int labelIndex = 7;     //5 values in each row of the iris.txt CSV: 4 input features followed by an integer label (class) index. Labels are the 5th value (index 4) in each row
        int numClasses = 3;     //3 classes (types of iris flowers) in the iris data set. Classes have integer values 0, 1 or 2
        int batchSize = 10000;    //Iris data set: 150 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses,1000000);

        final int numInputs = 7;
        int outputNum = 3;
        int iterations = 250;
        long seed = 1;

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .learningRate(0.1)
                .regularization(true).l2(1e-4)
                .list(3)
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(7)
                    .activation("tanh")
                    .weightInit(WeightInit.XAVIER)
                    .dropOut(0.66)
                    .build())
                .layer(1, new DenseLayer.Builder().nIn(7).nOut(7)
                    .activation("tanh")
                    .weightInit(WeightInit.XAVIER)
                    .dropOut(0.66)
                    .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .weightInit(WeightInit.XAVIER)
                    .activation("softmax")
                    .nIn(7).nOut(outputNum).build())
                    .backprop(true).pretrain(false)
                    .build();


        DataSet next = iterator.next();
        next.normalizeZeroMeanZeroUnitVariance();
        next.shuffle();
        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(0.80);  //Use 65% of data for training

        ListDataSetIterator train = new ListDataSetIterator(testAndTrain.getTrain().asList(), 1000);

        SplitTestAndTrain testAndValid = testAndTrain.getTest().splitTestAndTrain(0.50);

        ListDataSetIterator test = new ListDataSetIterator(testAndValid.getTrain().asList(), 1000);
        ListDataSetIterator valid = new ListDataSetIterator(testAndValid.getTest().asList(), 1000);

        String tempDir = System.getProperty("java.io.tmpdir");
        String exampleDirectory = FilenameUtils.concat(tempDir, "DL4JEarlyStoppingExample/");
        System.out.println("Location: " + exampleDirectory);

        EarlyStoppingModelSaver saver = new LocalFileModelSaver(exampleDirectory);
        EarlyStoppingConfiguration esConf = new EarlyStoppingConfiguration.Builder()
            .epochTerminationConditions(new ScoreImprovementEpochTerminationCondition(20)) //Max of 50 epochs
            .evaluateEveryNEpochs(1)
            .iterationTerminationConditions(new MaxTimeIterationTerminationCondition(150, TimeUnit.MINUTES)) //Max of 20 minutes
            .scoreCalculator(new DataSetLossCalculator(test, true))     //Calculate test set score
            .modelSaver(saver)
            .build();

        EarlyStoppingTrainer trainer = new EarlyStoppingTrainer(esConf,conf,train);

        //Conduct early stopping training:
        EarlyStoppingResult result = trainer.fit();
        System.out.println("Termination reason: " + result.getTerminationReason());
        System.out.println("Termination details: " + result.getTerminationDetails());
        System.out.println("Total epochs: " + result.getTotalEpochs());
        System.out.println("Best epoch number: " + result.getBestModelEpoch());
        System.out.println("Score at best epoch: " + result.getBestModelScore());

        //Print score vs. epoch
        Map<Integer,Double> scoreVsEpoch = result.getScoreVsEpoch();
        List<Integer> list = new ArrayList(scoreVsEpoch.keySet());
        Collections.sort(list);
        System.out.println("Score vs. Epoch:");
        for( Integer i : list){
            System.out.println(i + "\t" + scoreVsEpoch.get(i));
        }

        while(valid.hasNext()){
            DataSet validBatch = valid.next();
            Evaluation eval = new Evaluation(3);
            MultiLayerNetwork bestModel = (MultiLayerNetwork) result.getBestModel();
            INDArray output = bestModel.output(validBatch.getFeatureMatrix());
            eval.eval(validBatch.getLabels(), output);
            System.out.println(eval.stats());
        }
    }

}
