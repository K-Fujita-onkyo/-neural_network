/*

m5261108
KazukiFujita

Team project 1: Part 1
Delta learning rule

*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define I             3 //the number of data (include dammy input -1)
#define n_sample      4 //the number of sample
#define eta           0.5 //learning rate
#define lambda        1.0 
#define desired_error 0.01
#define sigmoid(x)    (2.0/(1.0+exp(-lambda*x))-1.0)
#define frand()       (rand()%10000/10001.0)
#define randomize()   srand((unsigned int)time(NULL))

double x[n_sample][I]={
  {0,0,-1},
  {0,1,-1},
  {1,0,-1},
  {1,1,-1}
};

double w[I];
double d[n_sample]={-1,-1,-1,1};
double o;

void Initialization(void);
void FindOutput(int);
void PrintResult(void);

int main(){
  int    i,p,q=0;
  double delta,Error=DBL_MAX;

  Initialization();//Initialize the weights at random
  while(Error>desired_error){
    q++;
    Error=0;
    for(p=0; p<n_sample; p++){
      FindOutput(p);
      Error+=0.5*pow(d[p]-o,2.0);
      for(i=0;i<I;i++){
	      delta=(d[p]-o)*(1-o*o)/2;
	      w[i]+=eta*delta*x[p][i];
      }
      //printf("Error in the %d-th learning cycle=%f\n",q,Error);
    } 
  }
  PrintResult();
}
  
/*************************************************************/
/* Initialization of the connection weights                  */
/*************************************************************/
void Initialization(void){
  int i;

  randomize();
  for(i=0; i<I; i++) w[i]=frand();
}

/*************************************************************/
/* Find the actual outputs of the network                    */
/*************************************************************/
void FindOutput(int p){
  int    i;
  double temp=0;

  for(i=0;i<I;i++) temp += w[i]*x[p][i];
  o = sigmoid(temp);
}

/*************************************************************/
/* Print out the final result                                */
/*************************************************************/
void PrintResult(void){
  int i,j;
  double u;

  printf("\n\n");
  printf("The connection weights of the neurons:\n");
  for(i=0;i<I;i++) printf("%5f ",w[i]);
  printf("\n\n");

  for(i=0;i<n_sample;i++){
    u=0;
    printf("Input: (");
    for(j=0;j<I;j++){
      printf("%1.0f", x[i][j]);
      if(j!=I-1)printf(",");
      u+=x[i][j]*w[j];
    }
    printf(") ");

    FindOutput(i);
    printf("f(u)=%f\n",o);
  }

}