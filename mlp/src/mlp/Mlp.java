package mlp;

//import static mlp.csvRead.readDataLineByLine;

import java.io.IOException;
import jxl.read.biff.BiffException;


public class Mlp
{  
    public static void main(String[] args) throws BiffException, IOException
    {
        NeuralNetwork nn=new NeuralNetwork();
        String file_path="F:/AcademicsAndProjects/MLP Project/MLP JAVA/iris.xls";
        FileInteraction read_file=new FileInteraction();
        float[][] input=read_file.getInputData(file_path, 4, 150);
        float[][] output=read_file.getOutputData(file_path,3,150);
        matrix mat_operation=new matrix();
        mat_operation.PrintMatrix(input);
        mat_operation.PrintMatrix(output);
        float[] error_per_epoch;
        error_per_epoch=nn.TrainNeuralNet(10,1, input, output);
        Predict pf=new Predict(4,nn.layer1.syn_wt,nn.layer2.syn_wt,error_per_epoch);
        java.awt.EventQueue.invokeLater(() -> {pf.setVisible(true);});
        
       
    }
    
}

