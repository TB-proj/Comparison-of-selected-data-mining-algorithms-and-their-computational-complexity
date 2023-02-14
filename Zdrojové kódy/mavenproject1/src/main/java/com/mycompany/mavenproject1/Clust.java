/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.mavenproject1;

import java.io.ByteArrayOutputStream;
import java.io.ObjectOutputStream;
import weka.core.Instances;
import java.lang.Runtime;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.clusterers.EM;
import weka.core.converters.ConverterUtils.DataSource;

public class Clust {

    public static void main(String[] args) throws Exception{


        // Start counting total memory usage
        double startMemory = (Runtime.getRuntime().totalMemory() -  Runtime.getRuntime().freeMemory())
                / 1024d / 1024d;
                
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ClusterEvaluation eTest;
        
        // Load dataset
        try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            DataSource source = new DataSource("C:/datasets/Generated/Clusters/Equal/200000-5-2clust.arff");
            Instances data = source.getDataSet();

        // Create a new clusterer    
            Clusterer cModel = (Clusterer)new EM();
            cModel.buildClusterer(data); 

        // Evaluate clusterer    
            eTest = new ClusterEvaluation();
            eTest.setClusterer(cModel);
            eTest.evaluateClusterer(data);
                       
            oos.writeObject(eTest);
        }
        
        // Print results     
        System.out.println(eTest.clusterResultsToString());
         
        // Print the size of memory of data structure
        System.out.println("size of data structure : " + baos.size() / 1024d / 1024d + " MB");

         // Count total memory usage
         double endMemory = (Runtime.getRuntime().totalMemory() -  Runtime.getRuntime().freeMemory())
            / 1024d / 1024d;
         
        // Print total memory usage
        System.out.println(" memory :" + (endMemory - startMemory));
    }
}




