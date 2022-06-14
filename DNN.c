#include <stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#pragma warning(disable:4996)

double x1, x2, output, target, err;
int neuron, neuron2, epoch;
double bw[15][1] = { {-0.5},{-0.5},{-0.5},{-0.5},{-0.5},{-0.5},{-0.5},{-0.5},{-0.5},{-0.5},{-0.5},{-0.5},{-0.5},{-0.5},{-0.5} };
double obw = -0.5;
double s = 0;
double s2 = 0;
double w[3][15];
double w1[3][15];
double w2[15][15];
double hiddenneuron1[15][1] = { {0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0} };
double hiddenneuron2[15][1] = { {0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0} };
double outputdelta = 0;
double seconddelta[15][1] = { {0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0} };
double ow[15][1] = { {-0.5},{-0.5},{-0.5},{-0.5},{-0.5},{-0.5},{-0.5},{-0.5},{-0.5},{-0.5},{-0.5},{-0.5},{-0.5},{-0.5},{-0.5} };
double outputneuron = 0;
double firstdelta[15][1] = { {0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0},{0} };



void Forward(double x1, double x2, double target) {
	for (int n = 0;n < neuron;n++) {
		s = x1 * w1[0][n] + x2 * w1[1][n] + w1[2][n];   ///hiddenlayer1

		hiddenneuron1[n][0] = 1 / (1 + exp(-s)); ///hidden layer1 u

	}


	for (int l = 0;l < neuron2;l++) {
		s2 = 0;
		for (int n = 0;n < neuron;n++) {
			s2 = s2 + hiddenneuron1[n][0] * w2[l][n];  ///hiddenlayer2
		}
		s2 = s2 + bw[l][0];
		hiddenneuron2[l][0] = 1 / (1 + exp(-s2)); ///hidden layer2 u
	}



	output = 0;
	for (int j = 0;j < neuron2;j++) {
		output = output + hiddenneuron2[j][0] * ow[j][0]; //outputlayer
	}
	output = output + 1 * obw;


	outputneuron = 1 / (1 + exp(-output)); //outputlayer u
}




void Back(double x1, double x2, double target) {

	outputdelta = outputneuron * (1 - outputneuron) * (target - outputneuron); //output delta
	for (int n = 0;n < neuron2;n++) {
		seconddelta[n][0] = hiddenneuron2[n][0] * (1 - hiddenneuron2[n][0]) * ow[n][0] * outputdelta;  //second layer delta
	}


	for (int n = 0;n < neuron;n++) {
		firstdelta[n][0] = 0;
		for (int l = 0;l < neuron2;l++) {
			firstdelta[n][0] = firstdelta[n][0] + (hiddenneuron1[n][0] * (1 - hiddenneuron1[n][0]) * w2[l][n] * seconddelta[l][0]);  //first layer delta
		}
	}




	for (int j = 0;j < neuron2;j++) {
		ow[j][0] = ow[j][0] + 0.5 * outputdelta * hiddenneuron2[j][0]; //output weight
	}
	for (int i = 0;i < neuron;i++) {
		w1[0][i] = w1[0][i] + 0.5 * firstdelta[i][0] * x1;   //first layer weight
		w1[1][i] = w1[1][i] + 0.5 * firstdelta[i][0] * x2;
		w1[2][i] = w1[2][i] + 0.5 * firstdelta[i][0] * 1; //first layer bias
	}
	for (int l = 0;l < neuron2;l++) {
		for (int n = 0;n < neuron;n++) {
			w2[l][n] = w2[l][n] + 0.5 * seconddelta[l][0] * hiddenneuron1[n][0]; //second layer weight

		}
	}

	for (int l = 0;l < neuron2;l++) {
		bw[l][0] = bw[l][0] + 0.5 * seconddelta[l][0] * 1; //second layer bias weight
	}


	obw = obw + 0.5 * outputdelta * 1; //output layer bias
}





double Error_back_propagation(double x1, double x2, double target) {
	Forward(x1, x2, target);
	Back(x1, x2, target);
}






void Gridtest(double x1, double x2) {
	for (int n = 0;n < neuron;n++) {
		s = x1 * w1[0][n] + x2 * w1[1][n] + w1[2][n];   ///hiddenlayer1

		hiddenneuron1[n][0] = 1 / (1 + exp(-s)); ///hiddenlayer1의 u
	}
	for (int l = 0;l < neuron2;l++) {
		s2 = 0;
		for (int n = 0;n < neuron;n++) {
			s2 = s2 + hiddenneuron1[n][0] * w2[l][n];  ///hiddenlayer2
		}
		s2 = s2 + bw[l][0];
		hiddenneuron2[l][0] = 1 / (1 + exp(-s2)); ///hiddenlayer2의 u
	}
	output = 0;
	for (int j = 0;j < neuron2;j++) {
		output = output + hiddenneuron2[j][0] * ow[j][0]; //outputlayer
	}
	output = output + 1 * obw;

	outputneuron = 1 / (1 + exp(-output)); //outputlayer의 u
}



int main() {

	double x1, x2, target = 0;
	double err = 0;

	printf("hidden layer1 neuron:\n");
	scanf("%d,&neuron");
	printf("hidden layer 2 neuron:\n");
	scanf("%d,&neuron2");
	printf("epoch:\n");
	scanf("%d,&epoch");




	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < neuron; j++) {
			w1[i][j] = (rand() % (20) - 10) * 0.1f;  ///w값 무작위 지정
			printf("w1[%d][%d]=%lf\n", i, j, w1[i][j]);
		}
	}

	for (int i = 0; i < neuron2; i++) {
		for (int j = 0; j < neuron; j++) {
			w2[i][j] = (rand() % (20) - 10) * 0.1f;
			printf("w2[%d][%d]=%lf\n", i, j, w2[i][j]);
		}
	}

	FILE* fp2 = fopen("Error.txt", "w");
	for (int i = 0;i < epoch;i++) {
		FILE* fp = fopen("input.txt", "r");
		err = 0;
		while (!((fscanf(fp, "%lf %lf %lf \n", &x1, &x2, &target)) == EOF)) {
			Error_back_propagation(x1, x2, target);

			err = err + fabs(target - outputneuron); //error 계산
		}
		fclose(fp);
		fprintf(fp2, "%d %lf\n", i, err);
	}
	fclose(fp2);

	FILE* fp3 = fopen("grid.txt", "w");//gridtest
	for (double i = -3.0;i < 3.0;i = i + 0.1) {
		for (double j = -3.0;j < 3.0;j = j + 0.1) {
			Gridtest(i, j);
			if (outputneuron > 0.5) {

				fprintf(fp3, "%lf %lf %d\n", i, j, 1);
			}
			else {
				fprintf(fp3, "%lf %lf %d\n", i, j, 0);
			}
		}
	}
	fclose(fp3);
	
	}
}
