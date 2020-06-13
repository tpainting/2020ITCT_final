#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#define PI 3.141592653589793238462643383
#define TEN7 10000000
#define TEN14 100000000000000

double x_0 = 0.0056, y_0 = 0.3678, xp_0 = 0.6229, yp_0 = 0.7676;
double mu = 0.8116;

void PrintMap(int* img, int m, int n) {
	//printf("%d %d\n", m, n);
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			printf("%d ", *(img+i*m+j));
		}
	}
    puts("");
}

void UpdateKey1(double s, double* x_bar0, double* y_bar0){
	double x_val = (x_0+(s+1)/(s+xp_0+yp_0+1));
	double y_val = (y_0+(s+2)/(s+xp_0+yp_0+2));
	*x_bar0 = x_val - (int)x_val;
	*y_bar0 = y_val - (int)y_val;
	return;
}

void UpdateKey2(double* xp_bar0, double* yp_bar0){
	double x_val = (xp_0+(1/(x_0+y_0+1)));
	double y_val = (yp_0+(2/(x_0+y_0+2)));
	*xp_bar0 = x_val - (int)x_val;
	*yp_bar0 = y_val - (int)y_val;
}

double* LASM2D(double x0, double y0, int ret_num){
	int skip_num = 200;
	int iter_num = ret_num/2 + ret_num%2;
	double xi = x0, yi = y0;
	double* ret_seq = (double*)malloc(sizeof(double)*ret_num);
	for(int i = 0; i < skip_num; i++){
		xi = sin(PI*mu*(yi+3)*xi*(1-xi));
		yi = sin(PI*mu*(xi+3)*yi*(1-yi));
	}
	int idx = 0;
	for(int i = 0; i < iter_num; i++){
		xi = sin(PI*mu*(yi+3)*xi*(1-xi));
		yi = sin(PI*mu*(xi+3)*yi*(1-yi));
		ret_seq[idx] = xi;
		idx++;
		if(idx >= ret_num) break;
		ret_seq[idx] = yi;
		idx++;
	}
    return ret_seq;
}

double Entropy(int* img, int m, int n, int start_col) {
	double test = log2(4);
	double entropy = 0.;
	int count[256];
	for(int i = 0; i < 256; i++){
		count[i] = 0;
	}
	for(int i = 0; i < m; i++){
		for(int j = start_col; j < n; j++){
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
bool Exist(int target, int* arr, int len){
	for(int i = 0; i < len; i++){
		if(target == arr[i]){
			return true;
		}
	}
	return false;
}
void Uniq(int* seq, int len){
	int min_num = 0;
	int exist[len], e_idx = 0;
	for(int i = 0; i < len; i++){
		bool is_exist = Exist(seq[i], exist, e_idx);
		if(!is_exist){
			exist[e_idx] = seq[i];
			e_idx++;
			if(seq[i] == min_num){
				while(true) {
					min_num++;
					is_exist = Exist(min_num, exist, e_idx);
					if(!is_exist) break;
				}
			}
		}else{
			seq[i] = min_num;
			exist[e_idx] = seq[i];
			e_idx++;
			while(true) {
				min_num++;
				is_exist = Exist(min_num, exist, e_idx);
				if(!is_exist) break;
			}
		}
	}
	return;
}
int* Encrypt(int* A, int m, int n){
	// Step 1
	double s = Entropy(A, m, n, 0);
	double x_bar0, y_bar0;
	UpdateKey1(s, &x_bar0, &y_bar0);
	double* P_seq = LASM2D(x_bar0, y_bar0, m*n);
	// Step 2
	int a = (int)ceil((x_0+y_0+1)*TEN7)%m;
	int b = (int)ceil((xp_0+yp_0+2)*TEN7)%n;
	int u[n], v[m];
	for(int i = 0; i < n; i++){
		u[i] = (int)((long)ceil(*(P_seq+a*n+i)*TEN14) % n);
	}
	for(int i = 0; i < m; i++){
		v[i] = (int)((long)ceil(*(P_seq+i*n+b)*TEN14) % m);
	}
	Uniq(u, n);
	Uniq(v, m);
	int B[m][n], tmp[m][n];
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			tmp[i][u[j]] = *(A+i*n+j);
		}
	}
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			B[v[i]][j] = tmp[i][j];
		}
	}
	//Step 3
	int R[m][n];
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			int w = (m*n+(i+1)+(j+1)) % 256;
			R[i][j] = (w + B[i][j]) % 256;
		}
	}
	//Step 4
	double xp_bar0, yp_bar0;
	UpdateKey2(&xp_bar0, &yp_bar0);
	double* K_seq = LASM2D(xp_bar0, yp_bar0, m*n);
	int K[m][n];
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			K[i][j] = (int)(((long)*(K_seq+i*m+j)*TEN14) % 256);
		}
	}
	//Step 5
	int* C = (int*)malloc(sizeof(int)*m*n);
	for(int i = 0; i < n; i++){
		int d;
		if(i < n-1){
			d = (int)((long)ceil(Entropy((int*)R, m, n, i+1)*TEN14) % n);
		}else{
			d = 0;
		}
		if(i == 0){
			for(int j = 0; j < m; j++){
				*(C+j*m+i) = (R[j][i]+(d+1)*K[j][i]+K[j][d]) % 256;
			}
		}else{
			for(int j = 0; j < m; j++){
				*(C+j*m+i) = (R[j][i]+(d+1)*(*(C+j*m+i-1))+(d+1)*K[j][i]+K[j][d]) % 256;
			}

		}
	}
	return C;
}

int img[2000][2000];
int main(int argc, char** argv){
	int m, n;
    //setvbuf(stdin, 0, 0, 0);
    //setvbuf(stdout, 0, 0, 0);
	while(~scanf("%d%d", &m, &n)){
		for(int i = 0; i < m; i++){
			for(int j = 0; j < n; j++){
				scanf("%d", &img[i][j]);
			}
		}
		int* C = Encrypt((int*)img, m, n);
		PrintMap(C, m, n);
	}
	return 0;
}
