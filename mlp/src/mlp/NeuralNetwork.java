package mlp;

public class NeuralNetwork
{
    NeuralLayer layer1;
    NeuralLayer layer2;
    
    NeuralNetwork()
    {
        
        this.layer1=new NeuralLayer(5, 4);            
        this.layer2=new NeuralLayer(3, 5);
        
        this.layer1.BuildWeightSynapse();
        this.layer2.BuildWeightSynapse();
        
        
    }
    void FeedForward(float[][]batch)
    {
        matrix mat_operation=new matrix();   
        
        this.layer1.WtSum(batch,batch.length,batch[0].length,1);        
        System.out.println("Calculated the weighted sum for layer 1");
        mat_operation.PrintMatrix(this.layer1.wt_sum);
        
        this.layer1.Activity();
        System.out.println("Calculated the activity  for layer 1");          
        mat_operation.PrintMatrix(this.layer1.activated_wt_sum);     
        
        this.layer2.WtSum(this.layer1.activated_wt_sum, this.layer1.activated_wt_sum.length,this.layer1.activated_wt_sum[0].length,1);
        System.out.println("Calculated the weighted sum for layer 2");          
        mat_operation.PrintMatrix(this.layer2.wt_sum);
        
        this.layer2.Activity();   
        System.out.println("Calculated the activity for layer 2");          
        mat_operation.PrintMatrix(this.layer2.activated_wt_sum);
    }
    
    float[][] BackPropagate(float[][] batch,float[][] output)
    {       
        matrix mat_operation=new matrix();               
       
        float layer2_error[][];
        layer2_error = mat_operation.Difference(output,this.layer2.activated_wt_sum);
        System.out.println("Calculated the error for layer 2"); 
        mat_operation.PrintMatrix(layer2_error);
        
        float layer2_delta[][];
        layer2_delta=mat_operation.multiplyElements(layer2_error,mat_operation.DerivativeActivity(this.layer2.activated_wt_sum));
        System.out.println("Calculated the delta for layer 2"); 
        mat_operation.PrintMatrix(layer2_delta);
        
        float layer1_error[][];
        layer1_error=mat_operation.multiplyMatrices(layer2_delta,mat_operation.Transpose(layer2.syn_wt), 0);
        System.out.println("Calculated the error for layer 1"); 
        mat_operation.PrintMatrix(layer1_error);
        
        float layer1_delta[][];
        layer1_delta=mat_operation.multiplyElements(layer1_error,mat_operation.DerivativeActivity(this.layer1.activated_wt_sum));
        System.out.println("Calculated the delta for layer 1"); 
        mat_operation.PrintMatrix(layer1_delta);
        
        float layer1_adjustment[][];
        layer1_adjustment=mat_operation.multiplyMatrices(mat_operation.Transpose(batch), layer1_delta, 0);
        System.out.println("Calculated the adjustment for layer 1"); 
        mat_operation.PrintMatrix(layer1_adjustment);
        
        float layer2_adjustment[][];
        layer2_adjustment=mat_operation.multiplyMatrices(mat_operation.Transpose(this.layer1.activated_wt_sum), layer2_delta,  0);
        System.out.println("Calculated the adjustment for layer 2"); 
        mat_operation.PrintMatrix(layer2_adjustment);
        
        
        this.layer1.updateWeight(layer1_adjustment);
        System.out.println("Updated W1"); 
        mat_operation.PrintMatrix(this.layer1.syn_wt);
        
        this.layer2.updateWeight(layer2_adjustment);
        System.out.println("Updated W2"); 
        mat_operation.PrintMatrix(this.layer2.syn_wt);
        return layer2_error;
        
    }
    
    float[] TrainNeuralNet(int epochs,int batch_size,float[][] input,float[][] output)
    {
        matrix mat_operation =new matrix();
        float cummulative_batch_error;
        float epoch_error=0;
        float[] error_per_epoch=new float[epochs];
        for(int i=0;i<epochs;i++)
        {
            System.out.println("Epoch:"+(i+1));
            int batch_no=0;
            while(batch_no<input.length/batch_size)
            {
                System.out.println("Batch:"+(batch_no+1));
                float batch_i[][]=new float[batch_size][input[0].length];
                float batch_o[][]=new float[batch_size][output[0].length];
                float[][] batch_error;
                
                int j=0;
                int x=batch_no*batch_size;
                while(j<batch_size && x<=input.length-batch_size)
                {
                    for(int k=0;k<input[0].length;k++)
                    {
                        batch_i[j][k]=input[x][j];
                    }
                    for(int k=0;k<output[0].length;k++)
                    {
                        batch_o[j][k]=output[x][j];
                    }
                    j++;
                    x++;
                }                
                FeedForward(batch_i);
                
                batch_error=BackPropagate(batch_i, batch_o);
                cummulative_batch_error=calculateErrror(batch_error);
                epoch_error = epoch_error+cummulative_batch_error;
                System.out.println("Synaptic Wt:Layer1");
                mat_operation.PrintMatrix(this.layer1.syn_wt);
                System.out.println("Synaptic Wt:Layer2");
                mat_operation.PrintMatrix(this.layer2.syn_wt);
                System.out.println("====================================================================");
                batch_no=batch_no+1;
            }
            epoch_error=epoch_error/batch_no;
            error_per_epoch[i]=epoch_error;
            epoch_error=0;
            
        }
        System.out.println("Done with Training!");
        System.out.println("Final Synaptic Wt:Layer1");
        mat_operation.PrintMatrix(this.layer1.syn_wt);
        System.out.println("Final Synaptic Wt:Layer2");
        mat_operation.PrintMatrix(this.layer2.syn_wt);       
        return error_per_epoch;
    }
    
   float calculateErrror(float[][] batch_error)
   {
        float cummulative_batch_error=0;
        float sum=0;
        for(int i=0;i<batch_error.length;i++)
        {
            for(int j=0;j<batch_error[0].length;j++)
            {
                sum=(float) (Math.pow(batch_error[i][j],2))+sum;
                
            }
            sum=sum/batch_error[0].length;
            sum=(float) Math.sqrt(sum);
            cummulative_batch_error=cummulative_batch_error+sum;
            sum=0;
            
        } 
        cummulative_batch_error=cummulative_batch_error/batch_error.length;
        return(cummulative_batch_error);
    }
    
   
    
    
   
    
}
