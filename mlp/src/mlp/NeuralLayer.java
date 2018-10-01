
package mlp;
public class NeuralLayer 
{
    int no_of_neurons;
    int no_of_inputs_per_neuron;
    float [][] syn_wt;
    float [][] wt_sum;
    float [][] activated_wt_sum;
   
    NeuralLayer(int no_of_neurons,int no_of_inputs_per_neuron)
    {
            this.no_of_neurons=no_of_neurons;
            this.no_of_inputs_per_neuron=no_of_inputs_per_neuron;
            this.syn_wt=null;
            this.wt_sum=null;
            this.activated_wt_sum=null;
            
    }
    void BuildWeightSynapse()
    {
            
            matrix mat_operation=new matrix(); 
            this.syn_wt = mat_operation.randomNormalizeMatrix(this.no_of_inputs_per_neuron,this.no_of_neurons);            
    }
    
    void WtSum(float [][]input,int rows,int columns,int bias)
    {
        matrix mat_operation=new matrix(); 
        this.wt_sum=mat_operation.multiplyMatrices(input,this.syn_wt,bias);
    }
    
    void Activity()
    {
        matrix mat_operation=new matrix();
        this.activated_wt_sum=mat_operation.activity(this.wt_sum);      
    }
    
    void updateWeight(float[][] adjustment)
    {
        matrix mat_operation=new matrix();
        this.syn_wt=mat_operation.Add(this.syn_wt,adjustment);
    }

}
