package mlp;

import java.io.FileInputStream;
import java.io.IOException;
import jxl.Sheet;
import jxl.Workbook;
import jxl.read.biff.BiffException;

public class FileInteraction
{
    private Workbook getWorkbook(String FilePath) throws BiffException, IOException
    {
        FileInputStream fs = new FileInputStream(FilePath);
        Workbook wb = Workbook.getWorkbook(fs);
        return(wb);
        
    }
    public float[][] getInputData(String FilePath,int no_of_features_per_input,int no_of_inputs) throws BiffException, IOException
    {
        
        Workbook wb=getWorkbook(FilePath);
       
        // TO get the access to the sheet
        Sheet sh = wb.getSheet("input");
       
        //declaring a matrix for storing the input
        
        float[][] input_matrix=new float[no_of_inputs][no_of_features_per_input];
        
        for(int i=0;i<no_of_inputs;i++)
        {
            for(int j=0;j<no_of_features_per_input;j++)
            {
                input_matrix[i][j]=Float.parseFloat(sh.getCell(j,i).getContents());
            }
        }
        return(input_matrix);
    }
    
     public float[][] getOutputData(String FilePath,int no_of_features_per_output,int no_of_outputs) throws BiffException, IOException
    {
        
        Workbook wb=getWorkbook(FilePath);
       
        // TO get the access to the sheet
        Sheet sh = wb.getSheet("output");
       
        //declaring a matrix for storing the input
        
        float[][] output_matrix=new float[no_of_outputs][no_of_features_per_output];
        
        for(int i=0;i<no_of_outputs;i++)
        {
            for(int j=0;j<no_of_features_per_output;j++)
            {
                
                output_matrix[i][j]=Float.parseFloat(sh.getCell(j,i).getContents());
            }
        }
        return(output_matrix);
    }
   
}