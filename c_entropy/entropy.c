#include <stdio.h>
#include <math.h>

double Entropy(int* img, int m, int n) {
	double test = log2(4);
	double entropy = 0.;
	int count[256];
	for(int i = 0; i < 256; i++){
		count[i] = 0;
	}
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			count[*(img+i*n+j)]++;
		}
	}
	double total = m*n;
	for(int i = 0; i < 256; i++){
		if(count[i] != 0) {
			double prob = count[i] / total;
			entropy += -prob*log2(prob);
		}
	}
	return entropy;
}

int main(int argc, char** argv) {
	int m, n;
	scanf("%d%d", &m, &n);
	int img[m][n];
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			scanf("%d", &img[i][j]);
		}
	}
	printf("%.20f\n", Entropy((int*)img, m, n));
	return 0;
}