package mlp;

import static java.lang.Math.exp;
import java.util.Random;

public class matrix 
{     
    public float[][] multiplyMatrices(float[][] firstMatrix, float[][] secondMatrix,int bias)
    {
        float sum=0;
        float[][] product = new float[firstMatrix.length][secondMatrix[0].length];
        for(int i = 0; i < firstMatrix.length; i++)
        {
            for (int j = 0; j < secondMatrix[0].length; j++)
            {
                for (int k = 0; k < firstMatrix[0].length; k++) 
                {
                    sum =sum + firstMatrix[i][k] * secondMatrix[k][j]+bias;
                }
                product[i][j]=sum;
                sum=0;
            }
        }
        return product;
    }
    
    public float[][] multiplyElements(float[][] firstMatrix, float[][] secondMatrix)
    {
        
        float[][] product = new float[firstMatrix.length][firstMatrix[0].length];
        for(int i = 0; i < firstMatrix.length; i++)
        {
            for (int j = 0; j < firstMatrix[0].length; j++)
            {
                product[i][j] =firstMatrix[i][j] * secondMatrix[i][j];             
            }
        }
        return product;
    }
    
    public float[][] randomNormalizeMatrix(int r,int c)
    {
        float [][] randomNormalMatrix=new float[r][c]; 
        for(int i = 0; i < r; i++)
        {
            for(int j = 0; j < c; j++)
            {    
                Random rand =new Random();
                randomNormalMatrix[i][j]= rand.nextFloat();
            }
         }
        return randomNormalMatrix;
        
    }
    
    public float[][] activity(float [][] matrix)
    {
        float [][] activityMatrix = new float[matrix.length][matrix[0].length]; 
        for(int i = 0; i < matrix.length; i++)
        {
            for(int j = 0; j < matrix[0].length; j++)
            {                            
                activityMatrix[i][j]=Sigmoid(matrix[i][j]);
            }
         }
        return activityMatrix;
        
    }
    
    public float[][] DerivativeActivity(float [][] matrix)
    {
        float [][] activityMatrix = new float[matrix.length][matrix[0].length]; 
        for(int i = 0; i < matrix.length; i++)
        {
            for(int j = 0; j < matrix[0].length; j++)
            {                            
                activityMatrix[i][j]=DerivativeSigmoid(matrix[i][j]);
            }
         }
        return activityMatrix;
        
    }

    public float Sigmoid(float x)
    {
        return(float) (1 / (1 + exp(-x)));
    }
    
    public float DerivativeSigmoid(float x)
    {
         return  (float) (exp(-x) / Math.pow((1 + exp(-x)),2));
    }
    
    float[][] Transpose(float[][] matrix)
    {
        float[][] transpose = new float[matrix[0].length][matrix.length];
        for(int i = 0; i < matrix.length; i++) 
        {
            for (int j = 0; j < matrix[0].length; j++) 
            {
                transpose[j][i] = matrix[i][j];
            }
        }
        return(transpose);
    }  
    
    float[][] Difference(float[][] matrix1,float[][] matrix2)
    {
        float[][] difference_matrix=new float[matrix1.length][matrix1[0].length];
        for(int i=0;i<matrix1.length;i++)
        {
            for(int j=0;j<matrix1[0].length;j++)
            {
                difference_matrix[i][j]=matrix1[i][j]-matrix2[i][j];
            }
        }
        return(difference_matrix);
    }
    
    float[][] Add(float[][] matrix1,float[][] matrix2)
    {
        float[][] addition_matrix=new float[matrix1.length][matrix1[0].length];
        for(int i=0;i<matrix1.length;i++)
        {
            for(int j=0;j<matrix1[0].length;j++)
            {
                addition_matrix[i][j]=matrix1[i][j]+matrix2[i][j];
            }
        }
        return(addition_matrix);
    }
    
    void PrintMatrix(float[][] matrix)
    {
        for (float[] matrix1 : matrix)
        {
            for (int j = 0; j<matrix[0].length; j++) 
            {
                System.out.print(matrix1[j] + "\t");
            }
            System.out.println("");
        }
    }
    
    float[][] MakeOutputMatrix(float[][] matrix)
    {
        float[][] output_matrix=new float[matrix.length][matrix[0].length];
        for(int i=0;i<matrix.length;i++)
        {
            for(int j=0;j<matrix[0].length;j++)
            {
                if(matrix[i][j]<=0.5)
                {
                    output_matrix[i][j]=0;
                }
                else
                {
                    output_matrix[i][j]=1;
                }
            }
        }
        return output_matrix;
    }

   
}
