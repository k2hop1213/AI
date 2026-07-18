/*
저번주차에 손계산한 결과와 동일한지 확인하기 위해
계산 편의상 사용했던 반올림 과정을 넣었습니다
반올림한 부분을 주석처리하고 반복횟수를 늘리면 가중치가 더욱 정교하게 업데이트 되고 오차가 줄어드는것을 확인할 수 있었습니다.

신경망의 구성을 자유롭게 하기 위해 동적할당을 적극 활용하여 코드를 짰습니다.
*/
/*
입력 설명
레이어의 개수와 각 레이어의 노드 수가 몇개 인지 입력 받습니다

입력 계층에 입력 값을 줍니다

사이즈에 맞게 가중치 행렬을 직접 입력합니다

편향과 목표값, 학습률, 반복횟수를 차례로 입력합니다
*/
/*
입력 예제
4
3 3 2 2

0.5 0.3 0.8

0.1 0.2 0.3
0.4 0.5 0.6
0.7 0.8 0.9

0.2 0.3 0.4
0.5 0.6 0.7

0.3 0.4
0.5 0.6

0.1

1.0 0.0

0.1

2
*/
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double sigmoid(double x) { //활성화 함수
    return 1.0 / (1.0 + exp(-x));

}

double sigmoid_deriv(double x) {
    return x * (1.0 - x);
}

void print_matrix(double** mat, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            printf("%lf ", mat[i][j]);
        }
        printf("\n");
    }
}

void free_matrix(double** mat, int row) { // 메모리 해제
    if (mat == NULL) return;
    for (int i = 0; i < row; i++) {
        free(mat[i]);
    }
    free(mat);
}

void multiply_matrix(double** a, double** b, double** c, int row1, int col1, int col2) { // 행렬의 곱 구현
    for (int i = 0; i < row1; i++) {
        for (int j = 0; j < col2; j++) {
            double temp = 0;
            for (int k = 0; k < col1; k++) {
                temp += a[i][k] * b[k][j];
            }
            c[i][j] = temp;
        }
    }
}

double** transpose(double** mat, int row,int col) { // 전치행렬을 반환하는 함수
    double** n = malloc(sizeof(double*) * col);
    for (int i = 0; i < col; i++) {
        n[i] = malloc(sizeof(double) * row);
    }
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            n[j][i] = mat[i][j];
        }
    }
    return n;
}



void forward(double*** MLP, double*** weight, int sizes[], int layer,double b) { // 순전파

    for (int i = 0; i < layer - 1; i++) {
        multiply_matrix(weight[i], MLP[i], MLP[i + 1], sizes[i + 1], sizes[i], 1);
        if (i != layer - 2) {
            for (int j = 0; j < sizes[i + 1]; j++) {
                MLP[i + 1][j][0] = sigmoid(MLP[i + 1][j][0] + b);
                MLP[i + 1][j][0] = round(MLP[i + 1][j][0] * 100.0) / 100.0;
            }
        }
        else {
            for (int j = 0; j < sizes[i + 1]; j++) {
                MLP[i + 1][j][0] += b; // 출력층에선 항등함수 사용
            }
        }
    }

}

void backward(double*** MLP, double*** weight,double*** trans_weight,double*** dw, double*** delta, double* b, double* target,int sizes[], int layer,double lr) { // 역전파
    double db = 0.0; // 편향의 기울기
    for (int i = 0; i < sizes[layer - 1]; i++) {
        delta[layer - 2][i][0] = -2.0 * (target[i] - MLP[layer - 1][i][0]); // 출력층에선 항등함수를 사용했기 때문
        delta[layer - 2][i][0] = round(delta[layer - 2][i][0] * 100.0) / 100.0;
        db += delta[layer - 2][i][0];
    }



    for (int i = layer - 2; i >= 1; i--) {
        multiply_matrix(trans_weight[i], delta[i], delta[i - 1], sizes[i], sizes[i + 1], 1);
        for (int j = 0; j < sizes[i]; j++) {
            delta[i - 1][j][0] *= sigmoid_deriv(MLP[i][j][0]);
            delta[i - 1][j][0] = round(delta[i - 1][j][0] * 100.0) / 100.0;
            db += delta[i - 1][j][0];
        }
    }
    for (int i = 0; i < layer - 1; i++) {
        double** temp = transpose(MLP[i], sizes[i], 1);
        multiply_matrix(delta[i], temp,  dw[i], sizes[i + 1], 1, sizes[i]);
        free_matrix(temp,1);
    }
    *b -= (lr * db);
    *b = round(*b * 100.0) / 100.0;
    for (int i = 0; i < layer - 1; i++) {
        for (int j = 0; j < sizes[i + 1]; j++) {
            for (int k = 0; k < sizes[i]; k++) {
                dw[i][j][k] = round(dw[i][j][k] * 100.0) / 100.0;
                weight[i][j][k] -= lr * dw[i][j][k];
                weight[i][j][k] = round(weight[i][j][k] * 100.0) / 100.0;
            }
        }
    }



}

int main() {
    double*** MLP = NULL; // 각 노드의 값들을 저장
    double*** weight = NULL; // 가중치
    double*** trans_weight = NULL; // 역전파에서 사용할 전치행렬
    double*** delta = NULL;
    double*** dw = NULL; // 기울기
    int layer, *sizes;//레이어 개수와 각 레이어의 노드 개수
    // layer - 1 : 가중치 행렬의 개수
    double b;// 편향
    double* target;// 목표값
    scanf("%d", &layer);


    MLP = malloc(sizeof(double**) * layer);

    weight = malloc(sizeof(double**) * (layer - 1));
    trans_weight = malloc(sizeof(double**) * (layer - 1));
    delta = malloc(sizeof(double**) * (layer - 1));
    dw = malloc(sizeof(double**) * (layer - 1));

    sizes = malloc(sizeof(int) * layer);

    for (int i = 0; i < layer; i++) {
        scanf("%d", &sizes[i]);
        MLP[i] = malloc(sizeof(double*) * sizes[i]);
        for (int j = 0; j < sizes[i]; j++) {
            MLP[i][j] = malloc(sizeof(double)); // 행렬곱을 하기위해서 2차원으로 만듦
        }
    }

    for (int i = 0; i < sizes[0]; i++) {
        scanf("%lf", &MLP[0][i][0]); // 입력층 입력
    }

    for (int i = 0; i < layer - 1; i++) {

        weight[i] = malloc(sizeof(double*) * sizes[i + 1]);
        dw[i] = malloc(sizeof(double*) * sizes[i + 1]);
        delta[i] = malloc(sizeof(double*) * sizes[i + 1]);

        for (int j = 0; j < sizes[i + 1]; j++) {

            weight[i][j] = malloc(sizeof(double) * sizes[i]);
            dw[i][j] = malloc(sizeof(double) * sizes[i]);

            delta[i][j] = malloc(sizeof(double));

            for (int k = 0; k < sizes[i]; k++) {
                scanf("%lf", &weight[i][j][k]); // 가중치 행렬 입력 받기
            }
        }
    }
    scanf("%lf", &b); // 편향
    
    target = malloc(sizeof(double) * (sizes[layer - 1]));
    for (int i = 0; i < sizes[layer - 1]; i++) {
        scanf("%lf", &target[i]);
    }
    double lr = 0.0;
    scanf("%lf", &lr);// 학습률
    double Error = 0.0;
    
    int rep = 0;
    // 반복횟수 입력받아서 원하는 횟수 만큼 순전파 역전파 반복하기
    

    scanf("%d", &rep);// 반복횟수
   


    for (int a = 1; a <= rep; a++) {
        printf("----------------------\n");

        printf("순전파 %d회차\n",a);
        forward(MLP, weight, sizes, layer, b);
        printf("입력레이어부터 출력레이어까지 차례대로 출력\n");

        if (a > 1) { // 전 회차에 할당한 공간이 있으므로 먼저 해제 시키기
            for (int i = 0; i < layer - 1; i++) {
                for (int j = 0; j < sizes[i]; j++) {
                    free(trans_weight[i][j]);
                }
                free(trans_weight[i]);
            }
        }

        for (int i = 0; i < layer - 1; i++) {
            trans_weight[i] = transpose(weight[i], sizes[i + 1], sizes[i]);
        }
        for (int i = 0; i < layer; i++) {

            print_matrix(MLP[i], sizes[i], 1);
            printf("\n");

        }
      

        Error = 0.0;
        for (int i = 0; i < sizes[layer - 1]; i++) {
            Error += pow(target[i] - MLP[layer - 1][i][0],2);
        }
        printf("오차 : %lf\n", Error);
        printf("----------------------\n");

        printf("역전파 %d회차\n",a);

        backward(MLP, weight, trans_weight, dw, delta, &b, target, sizes, layer, lr);
        printf("델타 값\n");
        for (int i = 0; i < layer - 1; i++) {
            print_matrix(delta[i], sizes[i + 1], 1);
            printf("\n");
        }
        printf("기울기 값\n");
        for (int i = 0; i < layer - 1; i++) {
            print_matrix(dw[i], sizes[i + 1], sizes[i]);
            printf("\n");
        }
        printf("업데이트 된 가중치\n");
        for (int i = 0; i < layer - 1; i++) {

            print_matrix(weight[i], sizes[i + 1], sizes[i]);
            printf("\n");
        }
        printf("업데이트 된 편향:%lf\n", b);
        printf("\n");
        printf("----------------------\n");
    }
    printf("----------------------\n");

    printf("순전파 %d회차\n", rep + 1);
    forward(MLP, weight, sizes, layer, b);
    printf("입력레이어부터 출력레이어까지 차례대로 출력\n");

    for (int i = 0; i < layer; i++) {

        print_matrix(MLP[i], sizes[i], 1);
        printf("\n");

    }
    Error = 0.0;
    for (int i = 0; i < sizes[layer - 1]; i++) {
        Error += pow(target[i] - MLP[layer - 1][i][0], 2);
    }
    printf("오차 : %lf\n", Error);
    printf("----------------------\n");
    
    // 동적 할당 해제

    for (int i = 0; i < layer; i++) {
        for (int j = 0; j < sizes[i]; j++) {
            free(MLP[i][j]);
        }
        free(MLP[i]);
    }
    free(MLP);


    for (int i = 0; i < layer - 1; i++) {
        for (int j = 0; j < sizes[i + 1]; j++) {
            free(weight[i][j]);
            free(dw[i][j]);
            free(delta[i][j]); 
        }
        free(weight[i]);
        free(dw[i]);
        free(delta[i]);
    }
    free(weight);
    free(dw);
    free(delta);

    for (int i = 0; i < layer - 1; i++) {
        for (int j = 0; j < sizes[i]; j++) {
            free(trans_weight[i][j]);
        }
        free(trans_weight[i]);
    }
    free(trans_weight);

    free(sizes);
    free(target);

    return 0;
}
