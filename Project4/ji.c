
#include <stdio.h>
#include <stdlib.h>

double GetRandom(double,double);

int main(void)
{
    int i;

    for (i = 0; i < 10; i++) {
        printf("%lf\n", GetRandom(4.3, 7.9));
    }

    return 0;
}

double GetRandom(double min, double max)
{
    return min + (rand() * (max - min + 1.0) / (1.0 + RAND_MAX));
}