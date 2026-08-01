#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define D 2   // 입력 차원
#define H 2   // 은닉 차원

void multiply_matrix(double* a, double* b, double* c, int row1, int col1, int col2) { 
    for (int i = 0; i < row1; i++) {
        for (int j = 0; j < col2; j++) {
            double temp = 0;
            for (int k = 0; k < col1; k++) {
                temp += a[i * col1 + k] * b[k * col2 + j];
            }
            c[i * col2 + j] = temp;
        }
    }
}

// h_next = tanh(Wx @ x + Wh @ h_prev + b_h)
void forward(double* Wx, double* Wh, double b_h, double* x, double* h_prev, double* h_next) {
    double wx_x[H];
    double wh_h[H];

    multiply_matrix(Wx, x, wx_x, H, D, 1);       
    multiply_matrix(Wh, h_prev, wh_h, H, H, 1);  

    for (int i = 0; i < H; i++) {
        h_next[i] = tanh(wx_x[i] + wh_h[i] + b_h);
    }
}

int main() {

    double Wx[2][2] = { {0.1,0.2},
                        {0.3,0.4} };
    double Wh[2][2] = { {0.5,0.0},
                        {0.0,0.5} };
    double b_h = 0.1;
    double Why[2][1] = { {0.5},
                          {0.5} };
    double b_y = 0.1;
    double x_t[3][2][1]
        = { {{1.0},
            {2.0}},
            {{0.0},
            {1.0}},
            {{1.0},
            {1.0}} };

    double h[H] = { 0.0, 0.0 };  // h_0 정의
    double h_next[H];

    double WhyT[1][2] = { { Why[0][0], Why[1][0] } };

    for (int t = 0; t < 3; t++) {
        forward((double*)Wx, (double*)Wh, b_h, (double*)x_t[t], h, h_next);

        for (int i = 0; i < H; i++) h[i] = h_next[i];

        double y;
        multiply_matrix((double*)WhyT, h, &y, 1, H, 1);   /* Why^T @ h_t -> 스칼라 */
        y += b_y;

        printf("t=%d  h=[%.4f, %.4f]  y=%.4f\n", t + 1, h[0], h[1], y);
    }

    return 0;
}
