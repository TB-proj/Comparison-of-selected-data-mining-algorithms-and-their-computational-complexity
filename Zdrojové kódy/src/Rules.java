
package com.mycompany.mavenproject1;

import java.io.ByteArrayOutputStream;
import java.io.ObjectOutputStream;
import weka.core.Instances;
import java.lang.Runtime;
import weka.associations.Apriori;
import weka.associations.AssociatorEvaluation;
import weka.core.converters.ConverterUtils.DataSource;

public class Rules {

    public static void main(String[] args) throws Exception{

        double startMemory = (Runtime.getRuntime().totalMemory() -  Runtime.getRuntime().freeMemory())
                / 1024d / 1024d;
        
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        AssociatorEvaluation eTest;
        try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            DataSource source = new DataSource("C:/datasets/Real/75000i.arff");
            Instances data = source.getDataSet();

            if (data.classIndex() == -1)
                data.setClassIndex(data.numAttributes() - 1);

            Apriori cModel = new Apriori();
            cModel.buildAssociations(data); 

            System.out.println(cModel);
            eTest = new AssociatorEvaluation();
            eTest.evaluate(cModel, data);
            String strSummary = eTest.toSummaryString();
            System.out.println(strSummary);
        }
            

         System.out.println("size of data structure : " + baos.size() / 1024d / 1024d + " MB");

         double endMemory = (Runtime.getRuntime().totalMemory() -  Runtime.getRuntime().freeMemory())
            / 1024d / 1024d;

        System.out.println(" memory :" + (endMemory - startMemory));
    }
}




