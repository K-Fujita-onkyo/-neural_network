/*
Kazuki Fujita: m5261108
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define I 4
#define M 3
#define P 150
#define alpha 0.1
#define n_update 20
#define INFTY 999999

void PrintResult(int);
void InitializeWeight();
double GetRandom(double,double);

double w[M][I];
double x[P][I];
double y[M];
double max[I]={-INFTY, -INFTY, -INFTY, -INFTY};
double min[I]={INFTY, INFTY, INFTY, INFTY};

//The main program
int main(){

	int m, m0, i, j, p, q;
	double norm, s, s0;

	//input iris data
	for(i=0;i<P;i++){
		for(j=0;j<I;j++){
			scanf("%lf",&x[i][j]);
			if(x[i][j]>max[j]) max[j]=x[i][j];
			if(x[i][j]<min[j]) min[j]=x[i][j];
		}
	}

	//Step 1
	//Initiarize weight
	InitializeWeight();
	PrintResult(0);

	/* Unsupervised learning */

	for (q = 0; q < n_update; q++){
		for (p = 0; p < P; p++){
			s0 = INFTY;

			//Step 2
			//Find the winner

			for (m = 0; m < M; m++){
				s = 0;
				for (i = 0; i < I; i++) s += pow(x[p][i]-w[m][i],2.0);
				s=sqrt(s);
				//Find the smallest error.
				if (s < s0){
					s0 = s;
					m0 = m;
				}
			}

			//Step 3
			//Update the weight of the winner
			for (i = 0; i < I; i++) {
				w[m0][i] += alpha * (x[p][i] - w[m0][i]);
			}
		}
		PrintResult(q);
	}

	/* Classify the training patterns */

	for (p = 0; p < P; p++){
		s0 = INFTY;
		for (m = 0; m < M; m++){
			s = 0;
			for (i = 0; i < I; i++) s += fabs(x[p][i] - w[m][i]);
			//printf("m=%d, s=%lf, s0=%lf\n", m, s, s0);
			if (s < s0){
				s0 = s;
				m0 = m;
			}
		}
		printf("Pattern[%d] belongs to %d-th class\n", p, m0);
	}
}

//Print out the result of the q-th iteration
void PrintResult(int q){
	int m, i;

	printf("\n\n");
	printf("Results in the %d-th iteration: \n", q);
	for (m = 0; m < M; m++)
	{
		for (i = 0; i < I; i++)
			printf("%5f ", w[m][i]);
		printf("\n");
	}
	printf("\n\n");
}

//Initialize weights
void InitializeWeight(){

	int m, i;

	for (m = 0; m < M; m++){
		for (i = 0; i < I; i++){
			w[m][i] = GetRandom(min[i],max[i]);
		}
	}
}

//Get random value that take from min to max
double GetRandom(double min, double max){
    return min + (rand() * (max - min + 1.0) / (1.0 + RAND_MAX));
}