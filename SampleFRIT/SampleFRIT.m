% FIRTのシミュレーション
clear;clc;

% 初期設定
Ts = 0.01;
A = [1 1; 0 -2];
B = [0; 1];
x0 = [0 0];
n = size(A, 1);
F_ini = [-0.8 2.0];
Hd = [
    tf(1, [1 -0.5 0], Ts);
    tf([1 -1], [1 -0.5 0], Ts)
];

% シミュレーション
k = 0:1:50; % ステップ数
N = length(k);
t = 0:Ts:(N-1)*Ts; % 時間ベクトル

% 短形波信号
v = zeros(1,N);
for index = 1:N
    if 1<index && index<7
        v(index) = 1;
    else
        v(index) = 0;
    end
end

% 状態の更新
x = zeros(N, 2);
x(1,:) = x0;
for index = 2:N
    u = F_ini * x(index-1,:)' + v(index-1);
    temp = A * x(index-1,:)' + B * u;
    x(index,:) = temp';
end

% 初期データ
x_ini = x;
u_ini = F_ini * x_ini' + v;

% 評価関数の最小化
Gamma = zeros(n*N, 1);
W = zeros(n*N, n);
for j = 1:n
    Gamma((j-1)*N+1:j*N) = x_ini(:, j) - lsim(Hd(j), u_ini, t);
    W((j-1)*N+1:j*N, :) = [lsim(Hd(j), x_ini(:, 1)', t) lsim(Hd(j), x_ini(:, 2)', t)];
end

% Gamma, W, W'*Wを.csvファイルに書き出す
csvwrite('Gamma.csv', Gamma);
csvwrite('W.csv', W);
csvwrite('Psi.csv', W'*W);

% 最適な状態フィードバックゲインの計算
F_ast = -Gamma' * W * inv(W' * W);

% 結果の表示
disp('F_ast = ');
disp(F_ast);
