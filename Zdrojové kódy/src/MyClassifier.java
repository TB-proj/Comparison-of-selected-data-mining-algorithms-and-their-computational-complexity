
package com.mycompany.mavenproject1;

import java.io.ByteArrayOutputStream;
import java.io.ObjectOutputStream;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import java.lang.Runtime;
import java.util.Random;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SimpleLogistic;
import weka.core.converters.ConverterUtils.DataSource;

public class MyClassifier {

    public static void main(String[] args) throws Exception{

        double startMemory = (Runtime.getRuntime().totalMemory() -  Runtime.getRuntime().freeMemory())
                / 1024d / 1024d;
        
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        Evaluation eTest;
        try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            DataSource source = new DataSource("C:/datasets/Generated/Classifiers/Equal/80000-30.arff");
            Instances data = source.getDataSet();

            if (data.classIndex() == -1)
                data.setClassIndex(data.numAttributes() - 1);

            Classifier cModel = (Classifier)new NaiveBayes();

            eTest = new Evaluation(data);
            eTest.crossValidateModel(cModel, data, 10, new Random(1));
            oos.writeObject(eTest);

        }
            
         String strSummary = eTest.toSummaryString();
         System.out.println(strSummary);
         System.out.println("size of data structure : " + baos.size() / 1024d / 1024d + " MB");

         double endMemory = (Runtime.getRuntime().totalMemory() -  Runtime.getRuntime().freeMemory())
            / 1024d / 1024d;

        System.out.println(" memory :" + (endMemory - startMemory));
    }
}




