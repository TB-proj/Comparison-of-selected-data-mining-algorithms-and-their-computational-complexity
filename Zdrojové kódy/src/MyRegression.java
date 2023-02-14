
package com.mycompany.mavenproject1;

import java.io.ByteArrayOutputStream;
import java.io.ObjectOutputStream;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import java.lang.Runtime;
import weka.classifiers.functions.LinearRegression;
import weka.core.converters.ConverterUtils.DataSource;

public class MyRegression {

    public static void main(String[] args) throws Exception{

        double startMemory = (Runtime.getRuntime().totalMemory() -  Runtime.getRuntime().freeMemory())
                / 1024d / 1024d;
        
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        Evaluation eTest;
        try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            DataSource source = new DataSource("C:/datasets/Real/houses.arff");
            DataSource sourceTest = new DataSource("C:/datasets/Real/houses.arff");
            Instances data = source.getDataSet();
            Instances dataTest = sourceTest.getDataSet();

            if (data.classIndex() == -1)
                data.setClassIndex(data.numAttributes() - 1);

            Classifier cModel = (Classifier)new LinearRegression();
            cModel.buildClassifier(data);  //should not be used when using cross validation

            eTest = new Evaluation(data);
            eTest.evaluateModel(cModel, dataTest);
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




