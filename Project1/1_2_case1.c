/*

m5261108
KazukiFujita

Team project 1: Part 2
Case1: discrete neurons

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>

#define N             3
#define R             3
#define n_sample      3
#define eta           0.5
#define lambda        1.0
#define desired_error 1.0
#define sigmoid(x)    (2.0/(1.0+exp(-lambda*x))-1.0)
#define frand()       (rand()%10000/10001.0)
#define randomize()   srand((unsigned int)time(NULL))

double x[n_sample][N]={
  {10,2,-1},
  {2,-5,-1},
  {-5,5,-1},
};
double d[n_sample][R]={
  {1,-1,-1},
  {-1,1,-1},
  {-1,-1,1},
};
double w[R][N];
double o[R];

void Initialization(void);
void FindOutput(int);
void PrintResult(void);
void PrintDesiredOutput(void);

int main(){
  int    i,j,p,q=0;
  double Error=DBL_MAX;
  double learningSignal;

  Initialization();
  while(Error>desired_error){
    q++;
    Error=0;
    for(p=0; p<n_sample; p++){
        FindOutput(p);
      for(i=0;i<R;i++){
	      Error+=0.5*pow(d[p][i]-o[i],2.0);
      }
      for(i=0;i<R;i++){
	      learningSignal=eta*(d[p][i]-o[i]);
        for(j=0;j<N;j++){
          w[i][j]+=learningSignal*x[p][j];
        }
      }
    } 
    //printf("Error in the %d-th learning cycle=%f\n",q,Error);
  }
  PrintResult();
}
  
/*************************************************************/
/* Initialization of the connection weights                  */
/*************************************************************/
void Initialization(void){
  int i,j;

  randomize();
  for(i=0;i<R;i++)
    for(j=0;j<N;j++)
      w[i][j]=frand()-0.5;
}

/*************************************************************/
/* Find the actual outputs of the network                    */
/*************************************************************/
void FindOutput(int p){
  int    i,j;
  double temp;

  for(i=0;i<R;i++){
    temp=0;
    for(j=0;j<N;j++){
      temp+=w[i][j]*x[p][j];
    }
    if(temp>0.0)o[i]=1.00;
    else o[i]=-1.00;
  }
}

/*************************************************************/
/* Print out the final result                                */
/*************************************************************/
void PrintResult(void){
  int i,j;
  
  printf("The connection weights are:\n\n");
  for(i=0;i<R;i++){
    for(j=0;j<N;j++)
      printf("%5f ",w[i][j]);
    printf("\n");
  }
  printf("\n\n");

  printf("Neuron outputs:\n\n");
  for(i=0;i<n_sample;i++){
    printf("Input: (");
    for(j=0;j<R;j++){
      if(j<R-1)printf("%f,",x[i][j]);
      else printf("%f)   f(u): ",x[i][j]);
    }

    FindOutput(i);

    printf("(");
    for(j=0;j<R;j++){
      if(j<R-1)printf("%f,",o[j]);
      else printf("%f",o[j]);
    }
    printf(")   ");

    PrintDesiredOutput();
  }
}


void PrintDesiredOutput(void){
  printf("Thus, the desired output is ");
  printf("(");
    for(int j=0;j<R;j++){
      if(o[j]>0)printf("1");
      else printf("-1");
      if(j<R-1)printf(",");
    }
    printf(")\n");
}
