#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define D 2   // 입력 차원
#define H 2   // 은닉 차원
#define T 3   // 시퀀스 길이

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

/* 시점별로 x, h_prev, h_next를 저장해둔다 (파이썬의 self.cache 역할, 시점마다 따로) */
double cache_x[T][D];
double cache_h_prev[T][H];
double cache_h_next[T][H];

void forward(double* Wx, double* Wh, double* b_h, double* x, double* h_prev, double* h_next, int t) {
    double wx_x[H];
    double wh_h[H];
    multiply_matrix(Wx, x, wx_x, H, D, 1);
    multiply_matrix(Wh, h_prev, wh_h, H, H, 1);
    for (int i = 0; i < H; i++) {
        h_next[i] = tanh(wx_x[i] + wh_h[i] + b_h[i]);
    }
    for (int i = 0; i < D; i++) cache_x[t][i] = x[i];
    for (int i = 0; i < H; i++) {
        cache_h_prev[t][i] = h_prev[i];
        cache_h_next[t][i] = h_next[i];
    }
}

/*
   t 시점의 역전파.
   dh_next: 미래(t+1)에서 넘어온 dh (없으면 0)
   y, target: 이 시점의 출력과 목표값 (dy = 2*(y-target))
   dh_prev_out: 이전(t-1) 시점으로 넘길 값 (출력 파라미터)
   dWx_sum, dWh_sum, db_h_sum, dWhy_sum, db_y_sum: 그래디언트 누적
*/
void backward(double* Wx, double* Wh, double* Why, int t, double y, double target,
    double* dh_next, double* dh_prev_out,
    double dWx_sum[H][D], double dWh_sum[H][H], double* db_h_sum ,double dWhy_sum[H], double* db_y_sum) 
{
    double* x = cache_x[t];
    double* h_prev = cache_h_prev[t];
    double* h_next = cache_h_next[t];

    double dy = 2.0 * (y - target);

    double dh[H];
    for (int i = 0; i < H; i++) dh[i] = dy * Why[i] + dh_next[i];

    double dt[H];
    for (int i = 0; i < H; i++) dt[i] = dh[i] * (1.0 - h_next[i] * h_next[i]);

    for (int i = 0; i < H; i++) {
        for (int j = 0; j < D; j++) dWx_sum[i][j] += dt[i] * x[j];
        for (int j = 0; j < H; j++) dWh_sum[i][j] += dt[i] * h_prev[j];
        dWhy_sum[i] += dy * h_next[i];
        db_h_sum[i] += dt[i];   /* b_h가 벡터이므로 각 원소별로 따로 누적 */
    }

    *db_y_sum += dy;

    for (int j = 0; j < H; j++) {
        double s = 0;
        for (int i = 0; i < H; i++) s += Wh[i * H + j] * dt[i];
        dh_prev_out[j] = s;
    }
}

void update(double* Wx, double* Wh, double* b_h, double* Why, double* b_y, double lr,
    double dWx_sum[H][D], double dWh_sum[H][H], double* db_h_sum, double dWhy_sum[H], double db_y_sum) {
    for (int i = 0; i < H; i++)
        for (int j = 0; j < D; j++)
            Wx[i * D + j] -= lr * dWx_sum[i][j];
    for (int i = 0; i < H; i++)
        for (int j = 0; j < H; j++)
            Wh[i * H + j] -= lr * dWh_sum[i][j];
    for (int i = 0; i < H; i++)
        b_h[i] -= lr * db_h_sum[i];
    for (int i = 0; i < H; i++)
        Why[i] -= lr * dWhy_sum[i];
    *b_y -= lr * db_y_sum;
}

int main() {
    double Wx[2][2] = { {0.1,0.2}, {0.3,0.4} };
    double Wh[2][2] = { {0.5,0.0}, {0.0,0.5} };
    double b_h[H] = { 0.1, 0.1 };
    double Why[2] = { 0.5, 0.5 };
    double b_y = 0.1;

    double x_t[3][2] = { {1.0,2.0}, {0.0,1.0}, {1.0,1.0} };

    double target = 0.0;
    double lr = 0.05;
    int rep = 3;

    for (int epoch = 0; epoch < rep; epoch++) {
        double h[H] = { 0.0, 0.0 };
        double ys[T];

        /* ---- 순전파 ---- */
        for (int t = 0; t < T; t++) {
            double h_next[H];
            forward((double*)Wx, (double*)Wh, b_h, x_t[t], h, h_next, t);

            for (int i = 0; i < H; i++) h[i] = h_next[i];

            double y = 0;
            for (int i = 0; i < H; i++) y += Why[i] * h[i];
            y += b_y;
            ys[t] = y;
        }

        double total_loss = 0;
        for (int t = 0; t < T; t++) total_loss += (target - ys[t]) * (target - ys[t]);
        printf("회차 %d  ys=[%.4f, %.4f, %.4f]  총 오차=%.4f\n", epoch + 1, ys[0], ys[1], ys[2], total_loss);

        if (epoch == rep - 1) break;  /* 마지막 회차는 순전파만 (역전파 rep-1회) */

        /* ---- 역전파 ---- */
        double dWx_sum[H][D] = { {0,0},{0,0} };
        double dWh_sum[H][H] = { {0,0},{0,0} };
        double db_h_sum[H] = { 0,0 };
        double dWhy_sum[H] = { 0,0 };
        double db_y_sum = 0;
        double dh_next[H] = { 0,0 };

        for (int t = T - 1; t >= 0; t--) {
            double dh_prev[H];
            backward((double*)Wx, (double*)Wh, Why, t, ys[t], target,
                dh_next, dh_prev, dWx_sum, dWh_sum, db_h_sum, dWhy_sum, &db_y_sum);
            for (int i = 0; i < H; i++) dh_next[i] = dh_prev[i];
        }

        /* ---- 가중치 업데이트 ---- */
        update((double*)Wx, (double*)Wh, b_h, Why, &b_y, lr,
            dWx_sum, dWh_sum, db_h_sum, dWhy_sum, db_y_sum);
    }

    return 0;
}
